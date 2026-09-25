"""Tests for generating JSON Schemas compatible with the Draft 7 dialect.

Schemas generated with `schema_dialect=DRAFT_07_SCHEMA_DIALECT` are cross-checked against
pydantic's own validation behavior using a local `jsonschema.Draft7Validator`.
"""

import jsonschema
import pytest
from pydantic_core import core_schema

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticUserError
from pydantic.json_schema import (
    DEFAULT_SCHEMA_DIALECT,
    DRAFT_07_SCHEMA_DIALECT,
    GenerateJsonSchema,
    models_json_schema,
)


def assert_draft_07_schema_matches_pydantic(schema, adapter, valid_samples, invalid_samples):
    """Assert that `schema` is a valid Draft 7 schema and that a local Draft7Validator
    accepts/rejects exactly the same samples as pydantic."""
    jsonschema.Draft7Validator.check_schema(schema)
    validator = jsonschema.Draft7Validator(schema)
    for sample in valid_samples:
        validator.validate(sample)
        adapter.validate_python(sample)
    for sample in invalid_samples:
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(sample)
        with pytest.raises(ValidationError):
            adapter.validate_python(sample)


def find_keys(value, key):
    """Recursively collect all values for `key` in a nested dict/list structure."""
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k == key:
                found.append(v)
            found.extend(find_keys(v, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(find_keys(item, key))
    return found


def test_default_dialect_is_unchanged():
    class Model(BaseModel):
        x: tuple[int, str]

    schema = Model.model_json_schema()
    assert GenerateJsonSchema().schema_dialect == DEFAULT_SCHEMA_DIALECT
    assert '$schema' not in schema
    assert schema['properties']['x']['prefixItems'] == [{'type': 'integer'}, {'type': 'string'}]


def test_draft_07_sets_schema_declaration():
    class Model(BaseModel):
        x: int

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert schema['$schema'] == 'http://json-schema.org/draft-07/schema#'
    jsonschema.Draft7Validator.check_schema(schema)


def test_draft_07_dialect_aliases():
    class Model(BaseModel):
        x: int

    for dialect in (
        'http://json-schema.org/draft-07/schema#',
        'http://json-schema.org/draft-07/schema',
        'https://json-schema.org/draft-07/schema#',
    ):
        schema = Model.model_json_schema(schema_dialect=dialect)
        assert schema['$schema'] == DRAFT_07_SCHEMA_DIALECT


def test_draft_07_dialect_on_generator_subclass():
    class Draft7GenerateJsonSchema(GenerateJsonSchema):
        schema_dialect = DRAFT_07_SCHEMA_DIALECT

    class Model(BaseModel):
        x: int

    schema = Model.model_json_schema(schema_generator=Draft7GenerateJsonSchema)
    assert schema['$schema'] == DRAFT_07_SCHEMA_DIALECT


def test_unsupported_dialect_raises():
    class Model(BaseModel):
        x: int

    with pytest.raises(PydanticUserError, match='Unsupported JSON Schema dialect') as exc_info:
        Model.model_json_schema(schema_dialect='https://json-schema.org/draft/2019-09/schema')
    assert exc_info.value.code == 'invalid-json-schema-dialect'


def test_recursive_model():
    class Node(BaseModel):
        name: str
        child: 'Node | None' = None

    schema = Node.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert not find_keys(schema, '$defs')
    assert 'Node' in schema['definitions']
    refs = find_keys(schema, '$ref')
    assert refs and all(ref.startswith('#/definitions/') for ref in refs)

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Node),
        valid_samples=[
            {'name': 'root'},
            {'name': 'root', 'child': None},
            {'name': 'root', 'child': {'name': 'a', 'child': {'name': 'b'}}},
        ],
        invalid_samples=[
            {},
            {'name': 1},
            {'name': 'root', 'child': {'name': 'a', 'child': {'name': 1}}},
            {'name': 'root', 'child': {'child': {'name': 'a'}}},
        ],
    )


def test_union_with_default():
    class Model(BaseModel):
        x: 'int | None' = None
        y: 'int | str' = 3

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert schema['properties']['x']['default'] is None
    assert schema['properties']['y']['default'] == 3

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Model),
        valid_samples=[{}, {'x': None}, {'x': 1, 'y': 'a'}, {'y': 2}],
        invalid_samples=[{'x': 'a'}, {'x': 1.5}, {'y': None}, {'y': ['a']}],
    )


def test_ref_to_model_with_default():
    class Child(BaseModel):
        x: int

    class Model(BaseModel):
        child: 'Child | None' = None

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Model),
        valid_samples=[{}, {'child': None}, {'child': {'x': 1}}],
        invalid_samples=[{'child': 1}, {'child': {'x': 'a'}}, {'child': {}}],
    )


def test_ref_with_adjacent_keywords_wrapped_in_all_of():
    class Child(BaseModel):
        x: int

    class Model(BaseModel):
        child: Child = Field(description='a child')

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    child_schema = schema['properties']['child']
    # In Draft 7, keywords adjacent to `$ref` are ignored, so the reference must be wrapped:
    assert child_schema == {'allOf': [{'$ref': '#/definitions/Child'}], 'description': 'a child'}

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Model),
        valid_samples=[{'child': {'x': 1}}],
        invalid_samples=[{'child': {'x': 'a'}}, {'child': None}, {}],
    )


def test_fixed_length_tuple():
    class Model(BaseModel):
        t: tuple[int, str]

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    tuple_schema = schema['properties']['t']
    assert 'prefixItems' not in tuple_schema
    assert tuple_schema['items'] == [{'type': 'integer'}, {'type': 'string'}]
    assert tuple_schema['minItems'] == 2
    assert tuple_schema['maxItems'] == 2

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Model),
        valid_samples=[{'t': [1, 'a']}],
        invalid_samples=[{'t': [1, 2]}, {'t': [1]}, {'t': [1, 'a', 3]}, {'t': 'a'}],
    )


def test_variadic_tuple():
    class Model(BaseModel):
        t: tuple[int, ...]

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    tuple_schema = schema['properties']['t']
    assert 'prefixItems' not in tuple_schema
    assert tuple_schema['items'] == {'type': 'integer'}

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Model),
        valid_samples=[{'t': []}, {'t': [1, 2, 3]}],
        invalid_samples=[{'t': ['a']}, {'t': [1, 'a']}],
    )


def test_fixed_prefix_with_variadic_items():
    class MixedTuple:
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            return core_schema.tuple_schema([core_schema.int_schema(), core_schema.str_schema()], variadic_item_index=1)

    adapter = TypeAdapter(MixedTuple)
    schema = adapter.json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    # `prefixItems` becomes the array form of `items`, and the schema for the extra
    # (variadic) items becomes `additionalItems`:
    assert schema == {
        '$schema': DRAFT_07_SCHEMA_DIALECT,
        'type': 'array',
        'items': [{'type': 'integer'}],
        'additionalItems': {'type': 'string'},
        'minItems': 1,
    }

    assert_draft_07_schema_matches_pydantic(
        schema,
        adapter,
        valid_samples=[[1], [1, 'a'], [1, 'a', 'b']],
        invalid_samples=[[], [[1]], [1, 2], [1, 'a', 2]],
    )


def test_nested_definitions():
    class Inner(BaseModel):
        x: int

    class Middle(BaseModel):
        inner: Inner
        inners: list[Inner]

    class Outer(BaseModel):
        middle: Middle

    schema = Outer.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert set(schema['definitions']) == {'Inner', 'Middle'}
    assert not find_keys(schema, '$defs')
    refs = find_keys(schema, '$ref')
    assert refs and all(ref.startswith('#/definitions/') for ref in refs)

    assert_draft_07_schema_matches_pydantic(
        schema,
        TypeAdapter(Outer),
        valid_samples=[{'middle': {'inner': {'x': 1}, 'inners': [{'x': 2}]}}],
        invalid_samples=[
            {'middle': {'inner': {'x': 'a'}, 'inners': []}},
            {'middle': {'inner': {'x': 1}, 'inners': [{'x': 'a'}]}},
            {'middle': {'inner': {'x': 1}}},
        ],
    )


def test_data_valued_keywords_are_not_rewritten():
    class Model(BaseModel):
        x: dict = Field(default={'prefixItems': [1]}, json_schema_extra={'examples': [{'prefixItems': []}]})

    schema = Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert schema['properties']['x']['default'] == {'prefixItems': [1]}
    assert schema['properties']['x']['examples'] == [{'prefixItems': []}]
    jsonschema.Draft7Validator.check_schema(schema)


def test_untranslatable_keyword_at_top_level():
    class Model(BaseModel):
        model_config = ConfigDict(json_schema_extra={'unevaluatedProperties': False})

        x: int

    with pytest.raises(PydanticInvalidForJsonSchema) as exc_info:
        Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    message = str(exc_info.value)
    assert "'unevaluatedProperties'" in message
    assert "path '$'" in message


def test_untranslatable_keyword_nested():
    class Model(BaseModel):
        x: int = Field(json_schema_extra={'unevaluatedItems': False})

    with pytest.raises(PydanticInvalidForJsonSchema) as exc_info:
        Model.model_json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    message = str(exc_info.value)
    assert "'unevaluatedItems'" in message
    assert "'$/properties/x'" in message


def test_models_json_schema_draft_07():
    class ModelA(BaseModel):
        a: 'ModelA | None' = None

    class ModelB(BaseModel):
        other: ModelA

    json_schemas_map, definitions_schema = models_json_schema(
        [(ModelA, 'validation'), (ModelB, 'validation')], schema_dialect=DRAFT_07_SCHEMA_DIALECT
    )

    assert definitions_schema['$schema'] == DRAFT_07_SCHEMA_DIALECT
    assert set(definitions_schema['definitions']) == {'ModelA', 'ModelB'}
    assert not find_keys(definitions_schema, '$defs')
    for schema in json_schemas_map.values():
        assert schema['$ref'].startswith('#/definitions/')
    jsonschema.Draft7Validator.check_schema(definitions_schema)


def test_type_adapter_json_schemas_draft_07():
    class Model(BaseModel):
        x: tuple[int, str]

    json_schemas_map, definitions_schema = TypeAdapter.json_schemas(
        [('Model', 'validation', TypeAdapter(Model))], schema_dialect=DRAFT_07_SCHEMA_DIALECT
    )

    assert definitions_schema['$schema'] == DRAFT_07_SCHEMA_DIALECT
    assert not find_keys(definitions_schema, '$defs')
    assert not find_keys(definitions_schema, 'prefixItems')
    jsonschema.Draft7Validator.check_schema(definitions_schema)


def test_type_adapter_json_schema_draft_07():
    adapter = TypeAdapter(list[int])
    schema = adapter.json_schema(schema_dialect=DRAFT_07_SCHEMA_DIALECT)
    assert schema['$schema'] == DRAFT_07_SCHEMA_DIALECT
    assert_draft_07_schema_matches_pydantic(
        schema,
        adapter,
        valid_samples=[[], [1, 2]],
        invalid_samples=[['a'], [1, 'a']],
    )
