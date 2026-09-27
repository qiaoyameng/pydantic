from __future__ import annotations

import operator
import sys
import types
import typing
from collections import ChainMap
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import reduce
from itertools import zip_longest
from types import prepare_class
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, TypeVar, cast
from weakref import WeakValueDictionary

import typing_extensions
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from . import _type_refs, _typing_extra
from ._forward_ref import PydanticRecursiveRef
from ._utils import all_identical, is_model_class

if TYPE_CHECKING:
    from ..main import BaseModel

GenericTypesCacheKey = tuple[Any, Any, tuple[Any, ...]]

# Note: We want to remove LimitedDict, but to do this, we'd need to improve the handling of generics caching.
#   Right now, to handle recursive generics, we some types must remain cached for brief periods without references.
#   By chaining the WeakValuesDict with a LimitedDict, we have a way to retain caching for all types with references,
#   while also retaining a limited number of types even without references. This is generally enough to build
#   specific recursive generic models without losing required items out of the cache.

KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100


class LimitedDict(dict[KT, VT]):
    def __init__(self, size_limit: int = _LIMITED_DICT_SIZE) -> None:
        self.size_limit = size_limit
        super().__init__()

    def __setitem__(self, key: KT, value: VT, /) -> None:
        super().__setitem__(key, value)
        if len(self) > self.size_limit:
            excess = len(self) - self.size_limit + self.size_limit // 10
            to_remove = list(self.keys())[:excess]
            for k in to_remove:
                del self[k]


# weak dictionaries allow the dynamically created parametrized versions of generic models to get collected
# once they are no longer referenced by the caller.
GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, 'type[BaseModel]']

if TYPE_CHECKING:

    class DeepChainMap(ChainMap[KT, VT]):  # type: ignore
        ...

else:

    class DeepChainMap(ChainMap):
        """Variant of ChainMap that allows direct updates to inner scopes.

        Taken from https://docs.python.org/3/library/collections.html#collections.ChainMap,
        with some light modifications for this use case.
        """

        def clear(self) -> None:
            for mapping in self.maps:
                mapping.clear()

        def __setitem__(self, key: KT, value: VT) -> None:
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: KT) -> None:
            hit = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)


# Despite the fact that LimitedDict _seems_ no longer necessary, I'm very nervous to actually remove it
# and discover later on that we need to re-add all this infrastructure...
# _GENERIC_TYPES_CACHE = DeepChainMap(GenericTypesCache(), LimitedDict())

_GENERIC_TYPES_CACHE = GenericTypesCache()


class PydanticGenericMetadata(TypedDict):
    origin: type[BaseModel] | None  # analogous to typing._GenericAlias.__origin__
    args: tuple[Any, ...]  # analogous to typing._GenericAlias.__args__
    parameters: tuple[TypeVar, ...]  # analogous to typing.Generic.__parameters__


def create_generic_submodel(
    model_name: str, origin: type[BaseModel], args: tuple[Any, ...], params: tuple[Any, ...]
) -> type[BaseModel]:
    """Dynamically create a submodel of a provided (generic) BaseModel.

    This is used when producing concrete parametrizations of generic models. This function
    only *creates* the new subclass; the schema/validators/serialization must be updated to
    reflect a concrete parametrization elsewhere.

    Args:
        model_name: The name of the newly created model.
        origin: The base class for the new model to inherit from.
        args: A tuple of generic metadata arguments.
        params: A tuple of generic metadata parameters.

    Returns:
        The created submodel.
    """
    namespace: dict[str, Any] = {'__module__': origin.__module__}
    # As per https://docs.python.org/3/reference/datamodel.html#slots:
    # "The action of a __slots__ declaration is not limited to the class where it is defined.
    # __slots__ declared in parents are available in child classes. However, instances of a
    # child subclass will get a __dict__ and __weakref__ unless the subclass also defines
    # __slots__".
    # Because when users parameterize a generic model, we create a subclass of such generic model
    # (what happens in this function), they can't control the fact that no slots is defined on the
    # dynamic subclass, and so even if they defined extra __slots__ on the generic class (which results
    # in the class *not* being weakref-able), the parameterized class *will* be weakref-able.
    # For this reason (and to make Pydantic generic models behavior closer to generic aliases),
    # we forward any slots from the origin:
    if '__slots__' in origin.__dict__:
        namespace['__slots__'] = origin.__dict__['__slots__']
    bases = (origin,)
    meta, ns, kwds = prepare_class(model_name, bases)
    namespace.update(ns)
    created_model = meta(
        model_name,
        bases,
        namespace,
        __pydantic_generic_metadata__={
            'origin': origin,
            'args': args,
            'parameters': params,
        },
        __pydantic_reset_parent_namespace__=False,
        **kwds,
    )

    model_module, called_globally = _get_caller_frame_info(depth=3)
    if called_globally:  # create global reference and therefore allow pickling
        object_by_reference = None
        reference_name = model_name
        reference_module_globals = sys.modules[model_module or created_model.__module__].__dict__
        while object_by_reference is not created_model:
            object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
            reference_name += '_'

    return created_model


def _get_caller_frame_info(depth: int = 2) -> tuple[str | None, bool]:
    """Used inside a function to check whether it was called globally.

    Args:
        depth: The depth to get the frame.

    Returns:
        A tuple contains `module_name` and `called_globally`.

    Raises:
        RuntimeError: If the function is not called inside a function.
    """
    try:
        previous_caller_frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be used inside another function') from e
    except AttributeError:  # sys module does not have _getframe function, so there's nothing we can do about it
        return None, False
    frame_globals = previous_caller_frame.f_globals
    return frame_globals.get('__name__'), previous_caller_frame.f_locals is frame_globals


DictValues: type[Any] = {}.values().__class__


def iter_contained_typevars(v: Any) -> Generator[TypeVar]:
    """Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found.

    This is inspired as an alternative to directly accessing the `__parameters__` attribute of a GenericAlias,
    since __parameters__ of (nested) generic BaseModel subclasses won't show up in that list.
    """
    if isinstance(v, TypeVar):
        yield v
    elif typing_objects.is_typevartuple(v):
        yield v  # type: ignore
    elif is_model_class(v):
        yield from v.__pydantic_generic_metadata__['parameters']
    elif isinstance(v, (DictValues, list, tuple)):
        for var in v:
            yield from iter_contained_typevars(var)
    else:
        args = get_args(v)
        for arg in args:
            yield from iter_contained_typevars(arg)


def get_args(v: Any) -> Any:
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('args')
    return typing_extensions.get_args(v)


def get_origin(v: Any) -> Any:
    pydantic_generic_metadata: PydanticGenericMetadata | None = getattr(v, '__pydantic_generic_metadata__', None)
    if pydantic_generic_metadata:
        return pydantic_generic_metadata.get('origin')
    return typing_extensions.get_origin(v)


def get_standard_typevars_map(cls: Any) -> dict[TypeVar, Any] | None:
    """Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the
    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.
    """
    origin = get_origin(cls)
    if origin is None:
        return None
    if not hasattr(origin, '__parameters__'):
        return None

    # In this case, we know that cls is a _GenericAlias, and origin is the generic type
    # So it is safe to access cls.__args__ and origin.__parameters__
    args: tuple[Any, ...] = cls.__args__  # type: ignore
    parameters: tuple[TypeVar, ...] = origin.__parameters__
    return _map_parameters_to_args(parameters, args)


def _map_parameters_to_args(parameters: tuple[Any, ...], args: tuple[Any, ...]) -> dict[TypeVar, Any]:
    """Zip generic parameters with their (fully materialized) arguments into a dictionary compatible
    with the `replace_types` function.

    If one of the parameters is a `TypeVarTuple`, it is mapped to a *tuple* of the arguments it
    consumes: parameters before it consume arguments from the front, parameters after it consume
    arguments from the back, and the `TypeVarTuple` consumes the remaining ones in the middle.
    """
    tvt_index = next((i for i, p in enumerate(parameters) if typing_objects.is_typevartuple(p)), None)
    if tvt_index is None:
        return dict(zip(parameters, args, strict=True))

    back_count = len(parameters) - tvt_index - 1
    split = len(args) - back_count
    typevars_map = dict(zip(parameters[:tvt_index], args[:tvt_index], strict=True))
    typevars_map[parameters[tvt_index]] = tuple(args[tvt_index:split])
    typevars_map.update(zip(parameters[tvt_index + 1 :], args[split:], strict=True))
    return typevars_map


def get_model_typevars_map(cls: type[BaseModel]) -> dict[TypeVar, Any]:
    """Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible
    with the `replace_types` function.

    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is
    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.
    """
    # TODO: This could be unified with `get_standard_typevars_map` if we stored the generic metadata
    #   in the __origin__, __args__, and __parameters__ attributes of the model.
    generic_metadata = cls.__pydantic_generic_metadata__
    origin = generic_metadata['origin']
    args = generic_metadata['args']
    if not args:
        # No need to go into `iter_contained_typevars`:
        return {}
    return _map_parameters_to_args(tuple(iter_contained_typevars(origin)), args)


def replace_types(type_: Any, type_map: Mapping[TypeVar, Any] | None) -> Any:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    Args:
        type_: The class or generic alias.
        type_map: Mapping from `TypeVar` instance to concrete types. `TypeVarTuple` instances
            can be mapped to a *tuple* of concrete types.

    Returns:
        A new type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced. If `type_` is an unpacked `TypeVarTuple`
        (i.e. `Unpack[Ts]`) mapped to a tuple of types, a tuple of the replaced types is
        returned instead; callers are expected to splice it into the surrounding type's
        arguments (see `replace_types_in_args`).

    Example:
        ```python
        from pydantic._internal._generics import replace_types

        replace_types(tuple[str, list[str] | float], {str: int})
        #> tuple[int, list[int] | float]
        ```
    """
    if not type_map:
        return type_

    type_args = get_args(type_)
    origin_type = get_origin(type_)

    if typing_objects.is_unpack(origin_type):
        # `Unpack[Ts]`, where `Ts` is a `TypeVarTuple` mapped to a tuple of types:
        unpacked = type_args[0] if type_args else None
        if unpacked is not None and unpacked in type_map:
            mapped = type_map[unpacked]
            items = mapped if isinstance(mapped, tuple) else (mapped,)
            return replace_types_in_args(items, type_map)

    if typing_objects.is_annotated(origin_type):
        annotated_type, *annotations = type_args
        annotated_type = replace_types(annotated_type, type_map)
        # TODO remove parentheses when we drop support for Python 3.10:
        return Annotated[(annotated_type, *annotations)]

    # Having type args is a good indicator that this is a typing special form
    # instance or a generic alias of some sort.
    if type_args:
        resolved_type_args = replace_types_in_args(type_args, type_map)
        if all_identical(type_args, resolved_type_args):
            # If all arguments are the same, there is no need to modify the
            # type or create a new object at all
            return type_

        if (
            origin_type is not None
            and isinstance(type_, _typing_extra.typing_base)
            and not isinstance(origin_type, _typing_extra.typing_base)
            and getattr(type_, '_name', None) is not None
        ):
            # In python < 3.9 generic aliases don't exist so any of these like `list`,
            # `type` or `collections.abc.Callable` need to be translated.
            # See: https://www.python.org/dev/peps/pep-0585
            origin_type = getattr(typing, type_._name)
        assert origin_type is not None

        if is_union_origin(origin_type):
            if any(typing_objects.is_any(arg) for arg in resolved_type_args):
                # `Any | T` ~ `Any`:
                resolved_type_args = (Any,)
            # `Never | T` ~ `T`:
            resolved_type_args = tuple(
                arg
                for arg in resolved_type_args
                if not (typing_objects.is_noreturn(arg) or typing_objects.is_never(arg))
            )

        # PEP-604 syntax (e.g. `list | str`) is represented with a types.UnionType object that does not
        # implement `__getitem__()`. In Python 3.14+, `typing.Union` and `types.UnionType` are the same,
        # and we instead rely on `typing.Union` as it implicitly converts string annotations to `ForwardRef`
        # instances (this is to avoid type errors as per https://github.com/python/cpython/pull/105366).
        if sys.version_info < (3, 14) and origin_type is types.UnionType:
            return reduce(operator.or_, resolved_type_args)
        # NotRequired[T] and Required[T] don't support tuple type resolved_type_args, hence the condition below
        return origin_type[resolved_type_args[0] if len(resolved_type_args) == 1 else resolved_type_args]

    # We handle pydantic generic models separately as they don't have the same
    # semantics as "typing" classes or generic aliases

    if not origin_type and is_model_class(type_):
        parameters = type_.__pydantic_generic_metadata__['parameters']
        if not parameters:
            return type_
        resolved_args: list[Any] = []
        for t in parameters:
            replaced = replace_types(t, type_map)
            if typing_objects.is_typevartuple(t) and isinstance(replaced, tuple):
                # A `TypeVarTuple` mapped to a tuple of types consumes a variable number of arguments:
                resolved_args.extend(replaced)
            else:
                resolved_args.append(replaced)
        resolved_type_args = tuple(resolved_args)
        if all_identical(parameters, resolved_type_args):
            return type_
        return type_[resolved_type_args]

    # Handle special case for typehints that can have lists as arguments.
    # `typing.Callable[[int, str], int]` is an example for this.
    if isinstance(type_, list):
        resolved_list = list(replace_types_in_args(tuple(type_), type_map))
        if all_identical(type_, resolved_list):
            return type_
        return resolved_list

    # If all else fails, we try to resolve the type directly and otherwise just
    # return the input with no modifications.
    return type_map.get(type_, type_)


def replace_types_in_args(type_args: tuple[Any, ...], type_map: Mapping[TypeVar, Any] | None) -> tuple[Any, ...]:
    """Apply `replace_types` to each item of a sequence of type arguments.

    Unlike a plain element-wise application, arguments of the form `Unpack[Ts]` (where `Ts`
    is a `TypeVarTuple` mapped to a tuple of types in `type_map`) are spliced into the
    resulting tuple, so that e.g. the arguments of `tuple[T, Unpack[Ts]]` are correctly
    resolved to the arguments of `tuple[int, str, bytes]` when `T` is mapped to `int` and
    `Ts` is mapped to `(str, bytes)`.
    """
    resolved_args: list[Any] = []
    for arg in type_args:
        replaced = replace_types(arg, type_map)
        if isinstance(replaced, tuple) and _typing_extra.unpack_type(arg) is not None:
            resolved_args.extend(replaced)
        else:
            resolved_args.append(replaced)
    return tuple(resolved_args)


def flatten_typevars_map_values(typevars_map: Mapping[TypeVar, Any]) -> tuple[Any, ...]:
    """Return the flattened type arguments of a parametrization, given the mapping between
    the parameters of a generic model and the arguments provided during parameterization.

    A `TypeVarTuple` parameter is mapped to a tuple of types, which is flattened here, so
    that e.g. parametrizing `Model[*Ts]` with `(int, str)` results in `(int, str)`.
    """
    return tuple(
        value
        for param, mapped in typevars_map.items()
        for value in (mapped if typing_objects.is_typevartuple(param) else (mapped,))
    )


def is_identity_parametrization(typevars_map: Mapping[TypeVar, Any]) -> bool:
    """Return whether the parametrization maps every parameter to itself.

    This is the case for e.g. `Model[T]` (or `Model[*Ts]` for variadic models), where
    no new submodel needs to be created.
    """
    for param, value in typevars_map.items():
        if typing_objects.is_typevartuple(param):
            if not (
                isinstance(value, tuple) and len(value) == 1 and _typing_extra.unpack_type(value[0]) is param
            ):
                return False
        elif param is not value:
            return False
    return True


def map_generic_model_arguments(cls: type[BaseModel], args: tuple[Any, ...]) -> dict[TypeVar, Any]:
    """Return a mapping between the parameters of a generic model and the provided arguments during parameterization.

    Raises:
        TypeError: If the number of arguments does not match the parameters (i.e. if providing too few or too many arguments).

    Example:
        ```python {test="skip" lint="skip"}
        class Model[T, U, V = int](BaseModel): ...

        map_generic_model_arguments(Model, (str, bytes))
        #> {T: str, U: bytes, V: int}

        map_generic_model_arguments(Model, (str,))
        #> TypeError: Too few arguments for <class '__main__.Model'>; actual 1, expected at least 2

        map_generic_model_arguments(Model, (str, bytes, int, complex))
        #> TypeError: Too many arguments for <class '__main__.Model'>; actual 4, expected 3
        ```

    Note:
        This function is analogous to the private `typing._check_generic_specialization` function.
    """
    parameters = cls.__pydantic_generic_metadata__['parameters']
    tvt_index = next((i for i, p in enumerate(parameters) if typing_objects.is_typevartuple(p)), None)
    if tvt_index is not None:
        return _map_variadic_arguments(cls, parameters, tvt_index, args)

    expected_len = len(parameters)
    typevars_map: dict[TypeVar, Any] = {}

    _missing = object()
    for parameter, argument in zip_longest(parameters, args, fillvalue=_missing):
        if parameter is _missing:
            raise TypeError(f'Too many arguments for {cls}; actual {len(args)}, expected {expected_len}')

        if argument is _missing:
            param = cast(TypeVar, parameter)
            try:
                has_default = param.has_default()  # pyright: ignore[reportAttributeAccessIssue]
            except AttributeError:
                # Happens if using `typing.TypeVar` (and not `typing_extensions`) on Python < 3.13.
                has_default = False
            if has_default:
                # The default might refer to other type parameters. For an example, see:
                # https://typing.python.org/en/latest/spec/generics.html#type-parameters-as-parameters-to-generics
                typevars_map[param] = replace_types(param.__default__, typevars_map)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                expected_len -= sum(hasattr(p, 'has_default') and p.has_default() for p in parameters)  # pyright: ignore[reportAttributeAccessIssue]
                raise TypeError(f'Too few arguments for {cls}; actual {len(args)}, expected at least {expected_len}')
        else:
            param = cast(TypeVar, parameter)
            typevars_map[param] = argument

    return typevars_map


def _has_default(param: Any) -> bool:
    try:
        return param.has_default()  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:
        # Happens if using `typing.TypeVar` (and not `typing_extensions`) on Python < 3.13.
        return False


def _map_variadic_arguments(
    cls: type[BaseModel], parameters: tuple[Any, ...], tvt_index: int, args: tuple[Any, ...]
) -> dict[TypeVar, Any]:
    """Return a mapping between the parameters of a generic model and the provided arguments,
    where one of the parameters (at index `tvt_index`) is a `TypeVarTuple`.

    Parameters before the `TypeVarTuple` consume arguments from the front, parameters after
    it consume arguments from the back, and the `TypeVarTuple` consumes whatever remains in
    the middle (possibly zero arguments). The `TypeVarTuple` is mapped to a *tuple* of types.
    """
    # Pad with a sentinel for parameters with a default that are reached exactly when
    # the provided arguments run out (mirrors how `typing` handles PEP 696 defaults):
    use_default = object()
    padded_args: list[Any] = list(args)
    for i, param in enumerate(parameters):
        if not typing_objects.is_typevartuple(param) and i == len(padded_args) and _has_default(param):
            padded_args.append(use_default)

    back_count = len(parameters) - tvt_index - 1
    if len(padded_args) < tvt_index + back_count:
        raise TypeError(
            f'Too few arguments for {cls}; actual {len(args)}, expected at least {len(parameters) - 1}'
        )

    split = len(padded_args) - back_count
    front_args = padded_args[:tvt_index]
    tvt_args = tuple(padded_args[tvt_index:split])
    back_args = padded_args[split:]

    tvt_param = parameters[tvt_index]
    if not tvt_args and _has_default(tvt_param):
        default = tvt_param.__default__
        unpacked_default = _typing_extra.unpack_type(default)
        if unpacked_default is not None:
            # e.g. `Unpack[tuple[int, str]]`:
            tvt_args = get_args(unpacked_default) or (unpacked_default,)
        else:
            tvt_args = tuple(default) if isinstance(default, tuple) else (default,)

    typevars_map: dict[TypeVar, Any] = {}
    for param, argument in zip(parameters[:tvt_index], front_args):
        typevars_map[param] = argument
    typevars_map[tvt_param] = tvt_args
    for param, argument in zip(parameters[tvt_index + 1 :], back_args):
        typevars_map[param] = argument

    # Resolve the defaults, which might refer to other type parameters. For an example, see:
    # https://typing.python.org/en/latest/spec/generics.html#type-parameters-as-parameters-to-generics
    for param, argument in list(typevars_map.items()):
        if argument is use_default:
            typevars_map[param] = replace_types(param.__default__, typevars_map)  # pyright: ignore[reportAttributeAccessIssue]

    return typevars_map


_generic_recursion_cache: ContextVar[set[str] | None] = ContextVar('_generic_recursion_cache', default=None)


@contextmanager
def generic_recursion_self_type(
    origin: type[BaseModel], args: tuple[Any, ...]
) -> Generator[PydanticRecursiveRef | None]:
    """This contextmanager should be placed around the recursive calls used to build a generic type,
    and accept as arguments the generic origin type and the type arguments being passed to it.

    If the same origin and arguments are observed twice, it implies that a self-reference placeholder
    can be used while building the core schema, and will produce a schema_ref that will be valid in the
    final parent schema.
    """
    previously_seen_type_refs = _generic_recursion_cache.get()
    if previously_seen_type_refs is None:
        previously_seen_type_refs = set()
        token = _generic_recursion_cache.set(previously_seen_type_refs)
    else:
        token = None

    try:
        type_ref = _type_refs.model_type_ref(origin, args_override=args)
        if type_ref in previously_seen_type_refs:
            self_type = PydanticRecursiveRef(type_ref=type_ref)
            yield self_type
        else:
            previously_seen_type_refs.add(type_ref)
            yield
            previously_seen_type_refs.remove(type_ref)
    finally:
        if token:
            _generic_recursion_cache.reset(token)


def recursively_defined_type_refs() -> set[str]:
    visited = _generic_recursion_cache.get()
    if not visited:
        return set()  # not in a generic recursion, so there are no types

    return visited.copy()  # don't allow modifications


def _generic_cache_get(key: GenericTypesCacheKey) -> type[BaseModel] | None:
    try:
        return _GENERIC_TYPES_CACHE.get(key)
    except TypeError:  # unhashable typevar values
        return None


def _generic_cache_set(key: GenericTypesCacheKey, value: type[BaseModel]) -> None:
    try:
        _GENERIC_TYPES_CACHE[key] = value
    except TypeError:  # unhashable typevar values
        pass


def get_cached_generic_type_early(parent: type[BaseModel], typevar_values: Any) -> type[BaseModel] | None:
    """The use of a two-stage cache lookup approach was necessary to have the highest performance possible for
    repeated calls to `__class_getitem__` on generic types (which may happen in tighter loops during runtime),
    while still ensuring that certain alternative parametrizations ultimately resolve to the same type.

    As a concrete example, this approach was necessary to make Model[List[T]][int] equal to Model[List[int]].
    The approach could be modified to not use two different cache keys at different points, but the
    _early_cache_key is optimized to be as quick to compute as possible (for repeated-access speed), and the
    _late_cache_key is optimized to be as "correct" as possible, so that two types that will ultimately be the
    same after resolving the type arguments will always produce cache hits.

    If we wanted to move to only using a single cache key per type, we would either need to always use the
    slower/more computationally intensive logic associated with _late_cache_key, or would need to accept
    that Model[List[T]][int] is a different type than Model[List[T]][int]. Because we rely on subclass relationships
    during validation, I think it is worthwhile to ensure that types that are functionally equivalent are actually
    equal.
    """
    return _generic_cache_get(_early_cache_key(parent, typevar_values))


def get_cached_generic_type_late(
    parent: type[BaseModel], typevar_values: Any, origin: type[BaseModel], args: tuple[Any, ...]
) -> type[BaseModel] | None:
    """See the docstring of `get_cached_generic_type_early` for more information about the two-stage cache lookup."""
    cached = _generic_cache_get(_late_cache_key(origin, args, typevar_values))
    if cached is not None:
        set_cached_generic_type(parent, typevar_values, cached, origin, args)
    return cached


def set_cached_generic_type(
    parent: type[BaseModel],
    typevar_values: tuple[Any, ...],
    type_: type[BaseModel],
    origin: type[BaseModel] | None = None,
    args: tuple[Any, ...] | None = None,
) -> None:
    """See the docstring of `get_cached_generic_type_early` for more information about why items are cached with
    two different keys.
    """
    _generic_cache_set(_early_cache_key(parent, typevar_values), type_)
    if len(typevar_values) == 1:
        _generic_cache_set(_early_cache_key(parent, typevar_values[0]), type_)
    if origin and args:
        _generic_cache_set(_late_cache_key(origin, args, typevar_values), type_)


def _union_orderings_key(typevar_values: Any) -> Any:
    """This is intended to help differentiate between Union types with the same arguments in different order.

    Thanks to caching internal to the `typing` module, it is not possible to distinguish between
    List[Union[int, float]] and List[Union[float, int]] (and similarly for other "parent" origins besides List)
    because `typing` considers Union[int, float] to be equal to Union[float, int].

    However, you _can_ distinguish between (top-level) Union[int, float] vs. Union[float, int].
    Because we parse items as the first Union type that is successful, we get slightly more consistent behavior
    if we make an effort to distinguish the ordering of items in a union. It would be best if we could _always_
    get the exact-correct order of items in the union, but that would require a change to the `typing` module itself.
    (See https://github.com/python/cpython/issues/86483 for reference.)
    """
    if isinstance(typevar_values, tuple):
        return tuple(_union_orderings_key(value) for value in typevar_values)
    elif typing_objects.is_union(typing_extensions.get_origin(typevar_values)):
        return get_args(typevar_values)
    else:
        return ()


def _early_cache_key(cls: type[BaseModel], typevar_values: Any) -> GenericTypesCacheKey:
    """This is intended for minimal computational overhead during lookups of cached types.

    Note that this is overly simplistic, and it's possible that two different cls/typevar_values
    inputs would ultimately result in the same type being created in BaseModel.__class_getitem__.
    To handle this, we have a fallback _late_cache_key that is checked later if the _early_cache_key
    lookup fails, and should result in a cache hit _precisely_ when the inputs to __class_getitem__
    would result in the same type.
    """
    return cls, typevar_values, _union_orderings_key(typevar_values)


def _late_cache_key(origin: type[BaseModel], args: tuple[Any, ...], typevar_values: Any) -> GenericTypesCacheKey:
    """This is intended for use later in the process of creating a new type, when we have more information
    about the exact args that will be passed. If it turns out that a different set of inputs to
    __class_getitem__ resulted in the same inputs to the generic type creation process, we can still
    return the cached type, and update the cache with the _early_cache_key as well.
    """
    # The _union_orderings_key is placed at the start here to ensure there cannot be a collision with an
    # _early_cache_key, as that function will always produce a BaseModel subclass as the first item in the key,
    # whereas this function will always produce a tuple as the first item in the key.
    return _union_orderings_key(typevar_values), origin, args
