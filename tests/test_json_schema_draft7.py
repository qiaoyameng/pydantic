"""Tests for generating JSON Schema Draft 7 compatible schemas via the `schema_dialect` argument."""

from typing import NamedTuple, Optional

import pytest
from jsonschema import Draft7Validator

from pydantic import BaseModel, Field, PydanticUserError, TypeAdapter, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import (
    JSON_SCHEMA_DIALECT_2020_12,
    JSON_SCHEMA_DIALECT_DRAFT_07,
    GenerateJsonSchema,
    models_json_schema,
)

DRAFT_07 = JSON_SCHEMA_DIALECT_DRAFT_07


def assert_no_draft_2020_12_keywords(node):
    """Recursively assert that a generated schema contains no Draft 2020-12-only constructs."""
    if isinstance(node, dict):
        assert '$defs' not in node
        assert 'prefixItems' not in node
        for key, value in node.items():
            if key == '$ref':
                assert '$defs' not in value
            else:
                assert_no_draft_2020_12_keywords(value)
    elif isinstance(node, list):
        for item in node:
            assert_no_draft_2020_12_keywords(item)


def assert_consistent_with_pydantic(model, schema, valid_samples, invalid_samples):
    """Check that Draft7Validator and pydantic agree on the given samples."""
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema)
    for sample in valid_samples:
        assert validator.is_valid(sample), f'Draft7Validator rejected valid sample: {sample!r}'
        model.model_validate(sample)
    for sample in invalid_samples:
        assert not validator.is_valid(sample), f'Draft7Validator accepted invalid sample: {sample!r}'
        with pytest.raises(ValidationError):
            model.model_validate(sample)


def test_default_dialect_is_unchanged():
    class Sub(BaseModel):
        x: int

    class Model(BaseModel):
        sub: Sub
        t: tuple[int, str]

    schema = Model.model_json_schema()
    assert '$schema' not in schema
    assert 'definitions' not in schema
    assert set(schema['$defs']) == {'Sub'}
    assert schema['properties']['sub'] == {'$ref': '#/$defs/Sub'}
    assert 'prefixItems' in schema['properties']['t']


def test_invalid_dialect_rejected():
    class Model(BaseModel):
        x: int

    with pytest.raises(PydanticUserError, match='Unsupported JSON schema dialect'):
        Model.model_json_schema(schema_dialect='http://json-schema.org/draft-04/schema#')

    with pytest.raises(PydanticUserError, match='Unsupported JSON schema dialect'):
        GenerateJsonSchema(schema_dialect='bogus')


def test_draft7_defs_and_refs_and_schema_declaration():
    class Inner(BaseModel):
        x: int

    class Middle(BaseModel):
        inner: Inner

    class Outer(BaseModel):
        middle: Middle

    schema = Outer.model_json_schema(schema_dialect=DRAFT_07)
    assert schema['$schema'] == 'http://json-schema.org/draft-07/schema#'
    assert set(schema['definitions']) == {'Inner', 'Middle'}
    assert schema['properties']['middle'] == {'$ref': '#/definitions/Middle'}
    assert schema['definitions']['Middle']['properties']['inner'] == {'$ref': '#/definitions/Inner'}
    assert_no_draft_2020_12_keywords(schema)
    Draft7Validator.check_schema(schema)


def test_draft7_ref_with_sibling_keywords_wrapped_in_all_of():
    class Sub(BaseModel):
        x: int

    class Model(BaseModel):
        sub: Sub = Field(description='a sub', default=None)

    schema = Model.model_json_schema(schema_dialect=DRAFT_07)
    sub_schema = schema['properties']['sub']
    # In Draft 7, siblings of `$ref` are ignored, so the ref must be wrapped in `allOf`
    # to keep the description/default metadata effective:
    assert '$ref' not in sub_schema
    assert sub_schema['allOf'] == [{'$ref': '#/definitions/Sub'}]
    assert sub_schema['description'] == 'a sub'
    Draft7Validator.check_schema(schema)


def test_draft7_recursive_model():
    class Node(BaseModel):
        name: str
        child: Optional['Node'] = None

    schema = Node.model_json_schema(schema_dialect=DRAFT_07)
    assert_no_draft_2020_12_keywords(schema)
    assert_consistent_with_pydantic(
        Node,
        schema,
        valid_samples=[
            {'name': 'a'},
            {'name': 'a', 'child': None},
            {'name': 'a', 'child': {'name': 'b', 'child': {'name': 'c'}}},
        ],
        invalid_samples=[
            {'name': 'a', 'child': {'name': 5}},
            {'name': 'a', 'child': {'child': None}},
            {'name': 'a', 'child': 5},
            {'child': None},
        ],
    )


def test_draft7_union_with_default():
    class Model(BaseModel):
        u: int | str = 5

    schema = Model.model_json_schema(schema_dialect=DRAFT_07)
    assert schema['properties']['u']['default'] == 5
    assert schema['properties']['u']['anyOf'] == [{'type': 'integer'}, {'type': 'string'}]
    assert_consistent_with_pydantic(
        Model,
        schema,
        valid_samples=[{}, {'u': 1}, {'u': 'a'}],
        invalid_samples=[{'u': None}, {'u': [1]}, {'u': {'k': 'v'}}],
    )


def test_draft7_fixed_length_tuple():
    class Model(BaseModel):
        t: tuple[int, str]

    schema = Model.model_json_schema(schema_dialect=DRAFT_07)
    t_schema = schema['properties']['t']
    assert t_schema['items'] == [{'type': 'integer'}, {'type': 'string'}]
    assert t_schema['minItems'] == 2
    assert t_schema['maxItems'] == 2
    assert 'prefixItems' not in t_schema
    assert_consistent_with_pydantic(
        Model,
        schema,
        valid_samples=[{'t': [1, 'a']}],
        invalid_samples=[
            {'t': []},
            {'t': [1]},
            {'t': [1, 'a', 3]},
            {'t': ['a', 'b']},
            {'t': [1, 2]},
        ],
    )


def test_draft7_variable_length_tuple():
    class Model(BaseModel):
        v: tuple[int, ...]

    schema = Model.model_json_schema(schema_dialect=DRAFT_07)
    v_schema = schema['properties']['v']
    assert v_schema['items'] == {'type': 'integer'}
    assert_consistent_with_pydantic(
        Model,
        schema,
        valid_samples=[{'v': []}, {'v': [1]}, {'v': [1, 2, 3]}],
        invalid_samples=[{'v': ['a']}, {'v': [1, 'a']}, {'v': None}],
    )


def test_draft7_named_tuple_with_defaults():
    class Pair(NamedTuple):
        a: int
        b: str = 'x'

    class Model(BaseModel):
        p: Pair

    schema = Model.model_json_schema(schema_dialect=DRAFT_07)
    p_schema = schema['definitions']['Pair']
    assert p_schema['items'] == [{'title': 'A', 'type': 'integer'}, {'default': 'x', 'title': 'B', 'type': 'string'}]
    assert p_schema['minItems'] == 1
    assert p_schema['maxItems'] == 2
    assert_consistent_with_pydantic(
        Model,
        schema,
        valid_samples=[{'p': [1]}, {'p': [1, 'a']}],
        invalid_samples=[{'p': []}, {'p': [1, 'a', 3]}, {'p': ['a']}],
    )


def test_draft7_prefix_items_with_additional_items_conversion():
    # `prefixItems` combined with a post-prefix `items` schema (as generated for variadic tuples
    # with a fixed prefix) must become a Draft 7 `items` array plus `additionalItems`.
    generator = GenerateJsonSchema(schema_dialect=DRAFT_07)
    source = {
        'type': 'array',
        'prefixItems': [{'type': 'integer'}],
        'items': {'type': 'string'},
        'minItems': 1,
    }
    converted = generator._convert_json_schema_to_draft_07(source)
    assert converted == {
        'type': 'array',
        'items': [{'type': 'integer'}],
        'additionalItems': {'type': 'string'},
        'minItems': 1,
    }
    Draft7Validator.check_schema(converted)
    validator = Draft7Validator(converted)
    assert validator.is_valid([1])
    assert validator.is_valid([1, 'a', 'b'])
    assert not validator.is_valid([1, 2])
    assert not validator.is_valid([])

    # `items: True` after `prefixItems` means "anything goes" and can simply be dropped:
    converted = generator._convert_json_schema_to_draft_07({'type': 'array', 'prefixItems': [], 'items': True})
    assert converted == {'type': 'array', 'items': []}


def test_draft7_dependent_keywords_conversion():
    generator = GenerateJsonSchema(schema_dialect=DRAFT_07)
    source = {
        'type': 'object',
        'dependentRequired': {'credit_card': ['billing_address']},
        'dependentSchemas': {'a': {'required': ['b']}},
    }
    converted = generator._convert_json_schema_to_draft_07(source)
    assert converted == {
        'type': 'object',
        'dependencies': {'credit_card': ['billing_address'], 'a': {'required': ['b']}},
    }
    Draft7Validator.check_schema(converted)


def test_draft7_unsupported_keyword_raises_with_path():
    class Model(BaseModel):
        x: list[int] = Field(json_schema_extra={'minContains': 2})

    with pytest.raises(PydanticInvalidForJsonSchema) as exc_info:
        Model.model_json_schema(schema_dialect=DRAFT_07)
    message = str(exc_info.value)
    assert 'minContains' in message
    assert '$.properties.x' in message

    # the same model generates fine with the default dialect
    assert Model.model_json_schema()['properties']['x']['minContains'] == 2


def test_draft7_unsupported_keyword_in_nested_definition():
    class Inner(BaseModel):
        x: int = Field(json_schema_extra={'unevaluatedItems': False})

    class Outer(BaseModel):
        inner: Inner

    with pytest.raises(PydanticInvalidForJsonSchema) as exc_info:
        Outer.model_json_schema(schema_dialect=DRAFT_07)
    message = str(exc_info.value)
    assert 'unevaluatedItems' in message
    assert '$.definitions.Inner.properties.x' in message


def test_draft7_models_json_schema():
    class Model(BaseModel):
        x: int

    schemas_map, definitions_doc = models_json_schema([(Model, 'validation')], schema_dialect=DRAFT_07)
    assert '$defs' not in definitions_doc
    assert set(definitions_doc['definitions']) == {'Model'}
    assert_no_draft_2020_12_keywords(definitions_doc)
    assert_no_draft_2020_12_keywords(schemas_map)


def test_draft7_type_adapter():
    adapter = TypeAdapter(list[int])
    schema = adapter.json_schema(schema_dialect=DRAFT_07)
    assert schema == {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'items': {'type': 'integer'},
        'type': 'array',
    }
    Draft7Validator.check_schema(schema)

    class Model(BaseModel):
        x: int

    schemas_map, definitions_doc = TypeAdapter.json_schemas(
        [(Model, 'validation', TypeAdapter(Model))], schema_dialect=DRAFT_07
    )
    assert set(definitions_doc['definitions']) == {'Model'}
    assert_no_draft_2020_12_keywords(definitions_doc)
    assert_no_draft_2020_12_keywords(schemas_map)


def test_draft7_explicit_default_dialect_keeps_legacy_output():
    class Model(BaseModel):
        x: int

    assert Model.model_json_schema(schema_dialect=JSON_SCHEMA_DIALECT_2020_12) == Model.model_json_schema()
