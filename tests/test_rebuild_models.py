import pytest

from pydantic import BaseModel, PydanticUndefinedAnnotation, TypeAdapter, ValidationError, rebuild_models


def create_mutually_dependent_models() -> tuple[type[BaseModel], type[BaseModel]]:
    class A(BaseModel):
        x: 'Alias'
        b: 'B | None' = None

    class B(BaseModel):
        y: 'Alias'
        a: 'A | None' = None

    return A, B


def test_rebuild_models_switches_aliased_type_atomically() -> None:
    A, B = create_mutually_dependent_models()
    assert not A.__pydantic_complete__
    assert not B.__pydantic_complete__

    # Duplicated roots and the circular reference between `A` and `B` are fine:
    assert rebuild_models([A, B, A], types_namespace={'A': A, 'B': B, 'Alias': int}) is True
    assert A.__pydantic_complete__
    assert B.__pydantic_complete__

    assert A(x=1, b={'y': 2}) == A(x=1, b=B(y=2))
    with pytest.raises(ValidationError):
        A(x='not an int')

    # Rebind the aliased type; both models must be rebuilt together:
    assert rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': str}) is True

    assert A(x='a', b={'y': 'b'}) == A(x='a', b=B(y='b'))
    with pytest.raises(ValidationError):
        A(x=1)
    with pytest.raises(ValidationError):
        B(y=1)

    json_schema = A.model_json_schema()
    assert json_schema['$defs']['A']['properties']['x'] == {'title': 'X', 'type': 'string'}
    assert json_schema['$defs']['B']['properties']['y'] == {'title': 'Y', 'type': 'string'}


def test_rebuild_models_existing_type_adapter_keeps_old_snapshot() -> None:
    A, B = create_mutually_dependent_models()
    rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': int})

    old_adapter = TypeAdapter(A)

    rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': str})

    # The existing adapter still uses the state at the time it was created:
    assert old_adapter.validate_python({'x': 1}).x == 1
    with pytest.raises(ValidationError):
        old_adapter.validate_python({'x': 'a'})

    # Newly created adapters see the new version:
    new_adapter = TypeAdapter(A)
    assert new_adapter.validate_python({'x': 'a'}).x == 'a'
    with pytest.raises(ValidationError):
        new_adapter.validate_python({'x': 1})


def test_rebuild_models_unresolvable_reference_keeps_previous_state() -> None:
    A, B = create_mutually_dependent_models()
    rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': str})

    old_fields_a = A.__pydantic_fields__
    old_validator_a, old_validator_b = A.__pydantic_validator__, B.__pydantic_validator__
    old_serializer_a, old_serializer_b = A.__pydantic_serializer__, B.__pydantic_serializer__
    old_core_schema_a, old_core_schema_b = A.__pydantic_core_schema__, B.__pydantic_core_schema__

    # `Alias` cannot be resolved in this namespace:
    with pytest.raises(PydanticUndefinedAnnotation, match="name 'Alias' is not defined"):
        rebuild_models([A, B], types_namespace={'A': A, 'B': B})

    # The whole group is left in its pre-rebuild (usable) state:
    assert A.__pydantic_fields__ is old_fields_a
    assert A.__pydantic_validator__ is old_validator_a
    assert B.__pydantic_validator__ is old_validator_b
    assert A.__pydantic_serializer__ is old_serializer_a
    assert B.__pydantic_serializer__ is old_serializer_b
    assert A.__pydantic_core_schema__ is old_core_schema_a
    assert B.__pydantic_core_schema__ is old_core_schema_b
    assert A.__pydantic_complete__
    assert B.__pydantic_complete__
    assert A(x='a', b={'y': 'b'}) == A(x='a', b=B(y='b'))


def test_rebuild_models_unresolvable_reference_raise_errors_false() -> None:
    A, B = create_mutually_dependent_models()
    rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': int})

    old_validator_a = A.__pydantic_validator__

    assert rebuild_models([A, B], types_namespace={'A': A, 'B': B}, raise_errors=False) is False

    assert A.__pydantic_validator__ is old_validator_a
    assert A.__pydantic_complete__
    assert A(x=1).x == 1


def test_rebuild_models_schema_generation_failure_keeps_previous_state() -> None:
    A, B = create_mutually_dependent_models()
    rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': int})

    old_validator_a, old_validator_b = A.__pydantic_validator__, B.__pydantic_validator__

    class NotAType:
        pass

    # `Alias` resolves, but no core schema can be generated for it:
    with pytest.raises(Exception, match='Unable to generate pydantic-core schema'):
        rebuild_models([A, B], types_namespace={'A': A, 'B': B, 'Alias': NotAType()})

    assert A.__pydantic_validator__ is old_validator_a
    assert B.__pydantic_validator__ is old_validator_b
    assert A.__pydantic_complete__
    assert B.__pydantic_complete__
    assert A(x=1, b={'y': 2}) == A(x=1, b=B(y=2))


def test_rebuild_models_invalid_input() -> None:
    with pytest.raises(TypeError, match='Expected a `BaseModel` subclass'):
        rebuild_models([int])  # type: ignore[list-item]
