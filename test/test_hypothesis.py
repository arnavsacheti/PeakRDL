import os
import unittest

from hypothesis import given, settings, assume
import hypothesis.strategies as st

from peakrdl.config import schema
from peakrdl.argfile import expand_tokens


# ---------------------------------------------------------------------------
# Strategies for generating schema objects
# ---------------------------------------------------------------------------

# Leaf schemas that don't require filesystem or imports
leaf_schemas = st.sampled_from([
    schema.String(),
    schema.Integer(),
    schema.Float(),
    schema.Boolean(),
    schema.AnyType(),
])


def schema_trees(max_leaves=8):
    """
    Recursively generate Schema trees: leaf types, Arrays, UserMappings,
    and FixedMappings with arbitrary nesting.
    """
    return st.recursive(
        leaf_schemas,
        lambda children: st.one_of(
            # Array of child schema
            children.map(lambda c: schema.Array(c)),
            # UserMapping with child value schema
            children.map(lambda c: schema.UserMapping(c)),
            # FixedMapping with 1-4 keys, each having a child schema
            st.dictionaries(
                st.text(
                    alphabet=st.characters(whitelist_categories=("L", "N")),
                    min_size=1,
                    max_size=8,
                ),
                children,
                min_size=1,
                max_size=4,
            ).map(lambda d: schema.FixedMapping(d)),
        ),
        max_leaves=max_leaves,
    )


_schema_key = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=8,
)


def fixed_mapping_schemas():
    """Generate FixedMapping schemas directly (no filtering needed)."""
    return st.dictionaries(
        _schema_key,
        schema_trees(),
        min_size=1,
        max_size=4,
    ).map(lambda d: schema.FixedMapping(d))


# ---------------------------------------------------------------------------
# 1. Path.extract always returns an absolute, normalized path
# ---------------------------------------------------------------------------
class TestPathNormalization(unittest.TestCase):
    @given(
        # Generate path-like strings: no null bytes (invalid in POSIX paths)
        path_str=st.text(
            alphabet=st.characters(blacklist_characters="\x00"),
        ),
        config_path=st.sampled_from([
            "/etc/peakrdl.toml",
            "/home/user/project/peakrdl.toml",
            "/peakrdl.toml",
        ]),
    )
    def test_result_is_always_absolute_and_normalized(self, path_str, config_path):
        p = schema.Path(shall_exist=False)
        result = p.extract(path_str, config_path, "test")

        self.assertTrue(
            os.path.isabs(result),
            f"Path.extract({path_str!r}) produced non-absolute path: {result!r}",
        )
        self.assertEqual(
            result,
            os.path.normpath(result),
            f"Path.extract({path_str!r}) produced non-normalized path: {result!r}",
        )


# ---------------------------------------------------------------------------
# 2. FixedMapping defaults are complete and type-correct
# ---------------------------------------------------------------------------
def _check_defaults(test_case, result, fm_schema):
    """
    Recursively verify that a FixedMapping extracted from {} has:
    - Exactly the expected keys
    - Correct default types for each sub-schema
    """
    test_case.assertEqual(
        set(result.keys()),
        set(fm_schema.schema.keys()),
        "Extracted keys don't match schema keys",
    )
    for key, sub_schema in fm_schema.schema.items():
        val = result[key]
        if isinstance(sub_schema, schema.Array):
            test_case.assertIsInstance(val, list)
            test_case.assertEqual(val, [])
        elif isinstance(sub_schema, schema.UserMapping):
            test_case.assertIsInstance(val, dict)
            test_case.assertEqual(val, {})
        elif isinstance(sub_schema, schema.FixedMapping):
            test_case.assertIsInstance(val, dict)
            # Recurse: nested FixedMapping should also have all keys defaulted
            _check_defaults(test_case, val, sub_schema)
        else:
            # All other schemas default to None
            test_case.assertIsNone(
                val,
                f"Expected None default for {type(sub_schema).__name__} "
                f"at key {key!r}, got {val!r}",
            )


class TestFixedMappingDefaults(unittest.TestCase):
    @given(fm=fixed_mapping_schemas())
    @settings(max_examples=200)
    def test_extract_empty_dict_has_all_keys_with_correct_defaults(self, fm):
        result = fm.extract({}, "/config.toml", "test")
        _check_defaults(self, result, fm)


# ---------------------------------------------------------------------------
# 3. Choice.extract accepts exactly the specified choices
# ---------------------------------------------------------------------------
class TestChoiceValidation(unittest.TestCase):
    @given(
        choices=st.lists(st.text(min_size=1), min_size=1, unique=True),
        data=st.data(),
    )
    def test_accepts_all_valid_choices(self, choices, data):
        """Every string in the choices list must be accepted."""
        c = schema.Choice(choices)
        s = data.draw(st.sampled_from(choices))
        result = c.extract(s, "/config.toml", "test")
        self.assertEqual(result, s)

    @given(
        choices=st.lists(st.text(min_size=1), min_size=1, unique=True),
        candidate=st.text(),
    )
    def test_rejects_invalid_choices(self, choices, candidate):
        """Any string NOT in the choices list must be rejected."""
        assume(candidate not in choices)
        c = schema.Choice(choices)
        with self.assertRaises(schema.SchemaException):
            c.extract(candidate, "/config.toml", "test")


# ---------------------------------------------------------------------------
# 4. expand_tokens is identity on token-free arguments
# ---------------------------------------------------------------------------
class TestExpandTokensIdentity(unittest.TestCase):
    @given(
        argv=st.lists(
            # Strings with no '$' character cannot trigger any token expansion
            st.text(
                alphabet=st.characters(blacklist_characters="$"),
            ),
        ),
        path=st.sampled_from([
            "/etc/argfile.f",
            "/home/user/project/args.f",
        ]),
    )
    def test_no_dollar_sign_means_identity(self, argv, path):
        result = expand_tokens(argv, path)
        self.assertEqual(result, argv)
