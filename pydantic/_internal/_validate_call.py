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
from ._typing_extra import eval_type, signature_no_eval


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


def _extract_async_gen_types(return_type: Any) -> tuple[Any, Any] | None:
    """Extract the `(yield_type, send_type)` pair from an async generator/iterator annotation.

    Returns `None` if `return_type` is not an `AsyncGenerator`/`AsyncIterator` annotation.
    `send_type` is `None` when no send type is annotated (i.e. `AsyncIterator[T]` or a bare
    `AsyncGenerator`), meaning sent values should be passed through untouched.
    """
    origin = get_origin(return_type)
    if origin is None:
        origin = return_type
    if not (inspect.isclass(origin) and issubclass(origin, collections.abc.AsyncIterator)):
        return None
    args = get_args(return_type)
    yield_type = args[0] if args else Any
    if len(args) > 1:
        # `collections.abc` aliases (unlike `typing` ones) don't convert `None` to `NoneType`
        send_type = type(None) if args[1] is None else args[1]
    else:
        send_type = None
    return yield_type, send_type


def _indexed_validation_error(exc: pydantic_core.ValidationError, title: str) -> pydantic_core.ValidationError:
    """Rebuild a `ValidationError` under a new title indicating which yield/send failed."""
    line_errors: list[pydantic_core.InitErrorDetails] = []
    for error in exc.errors():
        detail: pydantic_core.InitErrorDetails = {
            'type': error['type'],  # type: ignore
            'loc': error['loc'],
            'input': error['input'],
        }
        if 'ctx' in error:
            detail['ctx'] = error['ctx']
        line_errors.append(detail)
    return pydantic_core.ValidationError.from_exception_data(title, line_errors)


class _AsyncGenValidationWrapper:
    """An async iterator wrapper validating the yielded and sent values of an async generator.

    `athrow()`/`aclose()` (and thus task cancellation) are delegated straight to the wrapped
    generator, so its `finally` blocks still run. Yielded values are validated against the
    declared item type; values sent via `asend()` (after the first priming call) are validated
    against the declared send type, if any.
    """

    __slots__ = ('_gen', '_yield_validator', '_send_validator', '_qualname', '_yield_count', '_send_count', '_started')

    def __init__(
        self,
        gen: collections.abc.AsyncGenerator[Any, Any],
        yield_validator: pydantic_core.SchemaValidator | None,
        send_validator: pydantic_core.SchemaValidator | None,
        qualname: str,
    ) -> None:
        self._gen = gen
        self._yield_validator = yield_validator
        self._send_validator = send_validator
        self._qualname = qualname
        self._yield_count = 0
        self._send_count = 0
        self._started = False

    def __aiter__(self) -> _AsyncGenValidationWrapper:
        return self

    async def __anext__(self) -> Any:
        yielded = await self._gen.__anext__()
        self._started = True
        return self._validate_yield(yielded)

    async def asend(self, value: Any) -> Any:
        if self._started:
            value = self._validate_send(value)
        yielded = await self._gen.asend(value)
        self._started = True
        return self._validate_yield(yielded)

    async def athrow(self, *args: Any) -> Any:
        return self._validate_yield(await self._gen.athrow(*args))

    async def aclose(self) -> None:
        return await self._gen.aclose()

    def _validate_yield(self, value: Any) -> Any:
        self._yield_count += 1
        if self._yield_validator is None:
            return value
        try:
            return self._yield_validator.validate_python(value)
        except pydantic_core.ValidationError as exc:
            raise _indexed_validation_error(exc, f'{self._qualname} -> yield #{self._yield_count}') from exc

    def _validate_send(self, value: Any) -> Any:
        self._send_count += 1
        if self._send_validator is None:
            return value
        try:
            return self._send_validator.validate_python(value)
        except pydantic_core.ValidationError as exc:
            raise _indexed_validation_error(exc, f'{self._qualname} -> send #{self._send_count}') from exc


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
            if isinstance(return_type, str) and inspect.isasyncgenfunction(self.function):
                # `signature_no_eval` does not evaluate string annotations; resolve the return
                # annotation so that the async generator types below can be extracted from it.
                # If it can't be resolved yet, fall through and validate the (unresolved) annotation.
                try:
                    namespace = self.ns_resolver.types_namespace
                    return_type = eval_type(return_type, namespace.globals, namespace.locals)
                except NameError:
                    pass
            async_gen_types = (
                _extract_async_gen_types(return_type) if inspect.isasyncgenfunction(self.function) else None
            )
            if async_gen_types is not None:
                # Async generator function: only the call arguments are validated eagerly. Each
                # yielded/sent value is validated lazily, as the generator is iterated.
                yield_type, send_type = async_gen_types
                yield_validator = None if yield_type is Any else self._build_type_validator(yield_type, core_config)
                send_validator = (
                    None
                    if send_type is None or send_type is Any
                    else self._build_type_validator(send_type, core_config)
                )
                qualname = self.qualname

                def return_val_wrapper(gen: collections.abc.AsyncGenerator[Any, Any]) -> _AsyncGenValidationWrapper:
                    return _AsyncGenValidationWrapper(gen, yield_validator, send_validator, qualname)

                self.__return_pydantic_validator__ = return_val_wrapper
            elif inspect.iscoroutinefunction(self.function):
                validator = self._build_type_validator(return_type, core_config)

                async def return_val_wrapper(aw: Awaitable[Any]) -> None:
                    return validator.validate_python(await aw)

                self.__return_pydantic_validator__ = return_val_wrapper
            else:
                self.__return_pydantic_validator__ = self._build_type_validator(
                    return_type, core_config
                ).validate_python
        else:
            self.__return_pydantic_validator__ = None

        self.__pydantic_complete__ = True

    def _build_type_validator(self, type_: Any, core_config: dict[str, Any]) -> pydantic_core.SchemaValidator:
        gen_schema = GenerateSchema(self.config_wrapper, self.ns_resolver)
        schema = gen_schema.clean_schema(gen_schema.generate_schema(type_))
        return create_schema_validator(
            schema,
            self.schema_type,
            self.module,
            self.qualname,
            'validate_call',
            core_config,
            self.config_wrapper.plugin_settings,
        )

    def __call__(self, /, *args: Any, **kwargs: Any) -> Any:
        if not self.__pydantic_complete__:
            self._create_validators()

        res = self.__pydantic_validator__.validate_python(pydantic_core.ArgsKwargs(args, kwargs))
        if self.__return_pydantic_validator__:
            return self.__return_pydantic_validator__(res)
        else:
            return res
