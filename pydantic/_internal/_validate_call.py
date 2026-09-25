from __future__ import annotations as _annotations

import collections.abc
import functools
import inspect
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, get_args, get_origin

import pydantic_core

from ..config import ConfigDict
from ..plugin._schema_validator import create_schema_validator
from ._config import ConfigWrapper
from ._generate_schema import GenerateSchema, ValidateCallSupportedTypes
from ._namespace_utils import MappingNamespace, NsResolver, ns_for_function
from ._typing_extra import signature_no_eval, try_eval_type


def extract_function_name(func: ValidateCallSupportedTypes) -> str:
    """Extract the name of a `ValidateCallSupportedTypes` object."""
    return f'partial({func.func.__name__})' if isinstance(func, functools.partial) else func.__name__


def extract_function_qualname(func: ValidateCallSupportedTypes) -> str:
    """Extract the qualname of a `ValidateCallSupportedTypes` object."""
    return f'partial({func.func.__qualname__})' if isinstance(func, functools.partial) else func.__qualname__


def update_wrapper_attributes(wrapped: ValidateCallSupportedTypes, wrapper: Callable[..., Any]):
    """Update the `wrapper` function with the attributes of the `wrapped` function. Return the updated function."""
    if inspect.iscoroutinefunction(wrapped):

        @functools.wraps(wrapped)
        async def wrapper_function(*args, **kwargs):  # type: ignore
            return await wrapper(*args, **kwargs)
    else:

        @functools.wraps(wrapped)
        def wrapper_function(*args, **kwargs):
            return wrapper(*args, **kwargs)

    # We need to manually update this because `partial` object has no `__name__` and `__qualname__`.
    wrapper_function.__name__ = extract_function_name(wrapped)
    wrapper_function.__qualname__ = extract_function_qualname(wrapped)
    wrapper_function.raw_function = wrapped  # type: ignore

    return wrapper_function


def _indexed_validation_error(
    exc: pydantic_core.ValidationError, qualname: str, kind: str, index: int
) -> pydantic_core.ValidationError:
    """Rebuild a `ValidationError` so that every error is located at the `index`-th `kind` (`yield`/`send`) step."""
    line_errors: list[pydantic_core.InitErrorDetails] = []
    for error in exc.errors(include_url=False):
        details: pydantic_core.InitErrorDetails = {
            'type': error['type'],
            'loc': (kind, index, *error['loc']),
            'input': error['input'],
        }
        if 'ctx' in error:
            details['ctx'] = error['ctx']
        line_errors.append(details)
    return pydantic_core.ValidationError.from_exception_data(f'{qualname} {kind} #{index}', line_errors)


class _ValidatedAsyncGenerator:
    """An async generator wrapper validating yielded and sent values one at a time, as iteration happens.

    Delegation to the underlying async generator is direct (`__anext__`/`asend`/`athrow`/`aclose` are
    awaited on it), so `athrow`, `aclose` and task cancellation keep triggering the original
    generator's `finally` blocks.
    """

    __slots__ = ('_gen', '_item_validator', '_send_validator', '_qualname', '_started', '_yield_count', '_send_count')

    def __init__(
        self,
        gen: collections.abc.AsyncGenerator[Any, Any],
        item_validator: pydantic_core.SchemaValidator | None,
        send_validator: pydantic_core.SchemaValidator | None,
        qualname: str,
    ) -> None:
        self._gen = gen
        self._item_validator = item_validator
        self._send_validator = send_validator
        self._qualname = qualname
        self._started = False
        self._yield_count = 0
        self._send_count = 0

    def __aiter__(self) -> _ValidatedAsyncGenerator:
        return self

    async def __anext__(self) -> Any:
        item = await self._gen.__anext__()
        self._started = True
        self._yield_count += 1
        return self._validate_yield(item)

    async def asend(self, value: Any) -> Any:
        if self._started:
            # The first `asend()` only primes the generator (and must be `None` per the
            # iterator protocol, enforced by the underlying generator itself); only the
            # values sent afterwards are validated against the send type.
            value = self._validate_send(value)
        item = await self._gen.asend(value)
        self._started = True
        self._yield_count += 1
        return self._validate_yield(item)

    async def athrow(self, *args: Any) -> Any:
        item = await self._gen.athrow(*args)
        self._started = True
        self._yield_count += 1
        return self._validate_yield(item)

    async def aclose(self) -> None:
        return await self._gen.aclose()

    def _validate_yield(self, item: Any) -> Any:
        if self._item_validator is None:
            return item
        try:
            return self._item_validator.validate_python(item)
        except pydantic_core.ValidationError as exc:
            raise _indexed_validation_error(exc, self._qualname, 'yield', self._yield_count) from None

    def _validate_send(self, value: Any) -> Any:
        self._send_count += 1
        if self._send_validator is None:
            return value
        try:
            return self._send_validator.validate_python(value)
        except pydantic_core.ValidationError as exc:
            raise _indexed_validation_error(exc, self._qualname, 'send', self._send_count) from None


class ValidateCallWrapper:
    """This is a wrapper around a function that validates the arguments passed to it, and optionally the return value."""

    __slots__ = (
        'function',
        'validate_return',
        'schema_type',
        'module',
        'qualname',
        'ns_resolver',
        'config_wrapper',
        '__pydantic_complete__',
        '__pydantic_validator__',
        '__return_pydantic_validator__',
    )

    def __init__(
        self,
        function: ValidateCallSupportedTypes,
        config: ConfigDict | None,
        validate_return: bool,
        parent_namespace: MappingNamespace | None,
    ) -> None:
        self.function = function
        self.validate_return = validate_return
        if isinstance(function, partial):
            self.schema_type = function.func
            self.module = function.func.__module__
        else:
            self.schema_type = function
            self.module = function.__module__
        self.qualname = extract_function_qualname(function)

        self.ns_resolver = NsResolver(
            namespaces_tuple=ns_for_function(self.schema_type, parent_namespace=parent_namespace)
        )
        self.config_wrapper = ConfigWrapper(config)
        if not self.config_wrapper.defer_build:
            self._create_validators()
        else:
            self.__pydantic_complete__ = False

    def _create_validators(self) -> None:
        gen_schema = GenerateSchema(self.config_wrapper, self.ns_resolver)
        schema = gen_schema.clean_schema(gen_schema.generate_schema(self.function))
        core_config = self.config_wrapper.core_config(title=self.qualname)

        self.__pydantic_validator__ = create_schema_validator(
            schema,
            self.schema_type,
            self.module,
            self.qualname,
            'validate_call',
            core_config,
            self.config_wrapper.plugin_settings,
        )
        if self.validate_return:
            signature = signature_no_eval(self.function)
            return_type = signature.return_annotation if signature.return_annotation is not signature.empty else Any
            if inspect.isasyncgenfunction(self.function):
                self.__return_pydantic_validator__ = self._build_async_gen_return_wrapper(return_type, core_config)
            else:
                gen_schema = GenerateSchema(self.config_wrapper, self.ns_resolver)
                schema = gen_schema.clean_schema(gen_schema.generate_schema(return_type))
                validator = create_schema_validator(
                    schema,
                    self.schema_type,
                    self.module,
                    self.qualname,
                    'validate_call',
                    core_config,
                    self.config_wrapper.plugin_settings,
                )
                if inspect.iscoroutinefunction(self.function):

                    async def return_val_wrapper(aw: Awaitable[Any]) -> None:
                        return validator.validate_python(await aw)

                    self.__return_pydantic_validator__ = return_val_wrapper
                else:
                    self.__return_pydantic_validator__ = validator.validate_python
        else:
            self.__return_pydantic_validator__ = None

        self.__pydantic_complete__ = True

    def _build_async_gen_return_wrapper(
        self, return_type: Any, core_config: pydantic_core.CoreConfig
    ) -> Callable[[collections.abc.AsyncGenerator[Any, Any]], _ValidatedAsyncGenerator]:
        """Build a wrapper validating the items of the returned async generator as they are yielded,
        and the values sent into it (via `asend()`) as they are sent.

        The return annotation is interpreted as follows:

        * `AsyncGenerator[Y, S]`: yielded values are validated against `Y`, sent values against `S`.
        * `AsyncIterator[Y]`/`AsyncIterable[Y]` (or a `None` send type): only yielded values are
          validated, sent values are passed through untouched, per the iterator protocol.
        """
        globalns, localns = self.ns_resolver.types_namespace
        return_type, _ = try_eval_type(return_type, globalns, localns)
        origin = get_origin(return_type)
        args = get_args(return_type)

        send_type: Any = None
        if origin is collections.abc.AsyncGenerator:
            yield_type = args[0] if args else Any
            if len(args) > 1:
                send_type = args[1]
        elif origin in (collections.abc.AsyncIterator, collections.abc.AsyncIterable):
            yield_type = args[0] if args else Any
        else:
            yield_type = Any

        def build_validator(tp: Any) -> pydantic_core.SchemaValidator:
            gen_schema = GenerateSchema(self.config_wrapper, self.ns_resolver)
            schema = gen_schema.clean_schema(gen_schema.generate_schema(tp))
            return create_schema_validator(
                schema,
                self.schema_type,
                self.module,
                self.qualname,
                'validate_call',
                core_config,
                self.config_wrapper.plugin_settings,
            )

        item_validator = None if yield_type is Any else build_validator(yield_type)
        send_validator = (
            None if send_type is None or send_type is type(None) or send_type is Any else build_validator(send_type)
        )

        def wrap_async_gen(gen: collections.abc.AsyncGenerator[Any, Any]) -> _ValidatedAsyncGenerator:
            return _ValidatedAsyncGenerator(gen, item_validator, send_validator, self.qualname)

        return wrap_async_gen

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        if not self.__pydantic_complete__:
            self._create_validators()

        res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args, kwargs))
        if self.__return_pydantic_validator__:
            return self.__return_pydantic_validator__(res)
        else:
            return res
