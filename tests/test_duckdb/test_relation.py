import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import patito as pt
import polars as pl
import pytest
from typing_extensions import Literal

# Skip test module if DuckDB is not installed
if not pt._DUCKDB_AVAILABLE:
    pytest.skip("DuckDB not installed", allow_module_level=True)


def test_relation():
    """Test functionality of Relation class."""
    # Create a new in-memory database with dummy data
    db = pt.duckdb.Database()
    table_df = pl.DataFrame(
        {
            "column_1": [1, 2, 3],
            "column_2": ["a", "b", "c"],
        }
    )
    db.to_relation(table_df).create_table(name="table_name")
    table_relation = db.table("table_name")

    # A projection can be done in several different ways
    assert table_relation.select("column_1", "column_2") == table_relation.select(
        "column_1, column_2"
    )
    assert (
        table_relation.select("column_1, column_2")
        == table_relation[["column_1, column_2"]]
    )
    assert table_relation[["column_1, column_2"]] == table_relation
    assert table_relation.select("column_1") != table_relation.select("column_2")

    # We can also use kewyrod arguments to rename columns
    assert tuple(table_relation.select(column_3="column_1::varchar || column_2")) == (
        {"column_3": "1a"},
        {"column_3": "2b"},
        {"column_3": "3c"},
    )

    # The .get() method should only work if the filter matches a single row
    assert table_relation.get(column_1=1).column_2 == "a"

    # But raise if not exactly one matching row is found
    with pytest.raises(RuntimeError, match="Relation.get(.*) returned 0 rows!"):
        assert table_relation.get("column_1 = 4")
    with pytest.raises(RuntimeError, match="Relation.get(.*) returned 2 rows!"):
        assert table_relation.get("column_1 > 1")

    # The .get() should also accept a positional string
    assert table_relation.get("column_1 < 2").column_2 == "a"

    # And several positional strings
    assert table_relation.get("column_1 > 1", "column_1 < 3").column_2 == "b"

    # And a mix of positional and keyword arguments
    assert table_relation.get("column_1 < 2", column_2="a").column_2 == "a"

    # Order by statements shoud be respected when iterating over the relation
    assert tuple(table_relation.order("column_1 desc")) == (
        {"column_1": 3, "column_2": "c"},
        {"column_1": 2, "column_2": "b"},
        {"column_1": 1, "column_2": "a"},
    )

    # The plus operator acts as a union all
    assert (
        db.to_relation(table_df[:1])
        + db.to_relation(table_df[1:2])
        + db.to_relation(table_df[2:])
    ) == db.to_relation(table_df)

    # The union all must *not* remove duplicates
    assert db.to_relation(table_df) + db.to_relation(table_df) != db.to_relation(
        table_df
    )
    assert db.to_relation(table_df) + db.to_relation(table_df) == db.to_relation(
        pl.concat([table_df, table_df])
    )

    # You should be able to subscript columns
    assert table_relation["column_1"] == table_relation.select("column_1")
    assert table_relation[["column_1", "column_2"]] == table_relation

    # The relation's columns can be retrieved
    assert table_relation.columns == ["column_1", "column_2"]

    # You should be able to prefix and suffix all columns of a relation
    assert table_relation.add_prefix("prefix_").columns == [
        "prefix_column_1",
        "prefix_column_2",
    ]
    assert table_relation.add_suffix("_suffix").columns == [
        "column_1_suffix",
        "column_2_suffix",
    ]

    # You can drop one or more columns
    assert table_relation.drop("column_1").columns == ["column_2"]
    assert table_relation.select("*, 1 as column_3").drop(
        "column_1", "column_2"
    ).columns == ["column_3"]

    # You can rename columns
    assert set(table_relation.rename(column_1="new_name").columns) == {
        "new_name",
        "column_2",
    }

    # A value error must be raised if the source column does not exist
    with pytest.raises(
        ValueError,
        match=(
            "Column 'a' can not be renamed as it does not exist. "
            "The columns of the relation are: column_[12], column_[12]"
        ),
    ):
        table_relation.rename(a="new_name")

    # Null values should be correctly handled
    none_df = pl.DataFrame({"column_1": [1, None]})
    none_relation = db.to_relation(none_df)
    assert none_relation.filter("column_1 is null") == none_df.filter(
        pl.col("column_1").is_null()
    )

    # The .inner_join() method should work as INNER JOIN, not LEFT or OUTER JOIN
    left_relation = db.to_relation(
        pl.DataFrame(
            {
                "left_primary_key": [1, 2],
                "left_foreign_key": [10, 20],
            }
        )
    )
    right_relation = db.to_relation(
        pl.DataFrame(
            {
                "right_primary_key": [10],
            }
        )
    )
    joined_table = pl.DataFrame(
        {
            "left_primary_key": [1],
            "left_foreign_key": [10],
            "right_primary_key": [10],
        }
    )
    assert (
        left_relation.set_alias("l").inner_join(
            right_relation.set_alias("r"),
            on="l.left_foreign_key = r.right_primary_key",
        )
        == joined_table
    )

    # But the .left_join() method performs a LEFT JOIN
    left_joined_table = pl.DataFrame(
        {
            "left_primary_key": [1, 2],
            "left_foreign_key": [10, 20],
            "right_primary_key": [10, None],
        }
    )
    assert (
        left_relation.set_alias("l").left_join(
            right_relation.set_alias("r"),
            on="l.left_foreign_key = r.right_primary_key",
        )
        == left_joined_table
    )


def test_star_select():
    """It should select all columns with star."""
    df = pt.DataFrame({"a": [1, 2], "b": [3, 4]})
    relation = pt.duckdb.Relation(df)
    assert relation.select("*") == relation


def test_casting_relations_between_database_connections():
    """It should raise when you try to mix databases."""
    db_1 = pt.duckdb.Database()
    relation_1 = db_1.query("select 1 as a")
    db_2 = pt.duckdb.Database()
    relation_2 = db_2.query("select 1 as a")
    with pytest.raises(
        ValueError,
        match="Relations can't be casted between database connections.",
    ):
        relation_1 + relation_2  # pyright: ignore


def test_creating_relation_from_pandas_df():
    """It should be able to create a relation from a pandas dataframe."""
    pd = pytest.importorskip("pandas")
    pandas_df = pd.DataFrame({"a": [1, 2]})
    relation = pt.duckdb.Relation(pandas_df)
    pd.testing.assert_frame_equal(relation.to_pandas(), pandas_df)


def test_creating_relation_from_a_csv_file(tmp_path):
    """It should be able to create a relation from a CSV path."""
    df = pl.DataFrame({"a": [1, 2]})
    csv_path = tmp_path / "test.csv"
    df.write_csv(csv_path)
    relation = pt.duckdb.Relation(csv_path)
    assert relation.to_df().frame_equal(df)


def test_creating_relation_from_a_parquet_file(tmp_path):
    """It should be able to create a relation from a parquet path."""
    df = pl.DataFrame({"a": [1, 2]})
    parquet_path = tmp_path / "test.parquet"
    df.write_parquet(parquet_path, compression="uncompressed")
    relation = pt.duckdb.Relation(parquet_path)
    assert relation.to_df().frame_equal(df)


def test_creating_relation_from_a_unknown_file_format(tmp_path):
    """It should raise when you try to create relation from unknown path."""
    with pytest.raises(
        ValueError,
        match="Unsupported file suffix '.unknown' for data import!",
    ):
        pt.duckdb.Relation(Path("test.unknown"))

    with pytest.raises(
        ValueError,
        match="Unsupported file suffix '' for data import!",
    ):
        pt.duckdb.Relation(Path("test"))


def test_relation_with_default_database():
    """It should be constructable with the default DuckDB cursor."""
    import duckdb

    relation_a = pt.duckdb.Relation("select 1 as a")
    assert relation_a.database.connection is duckdb.default_connection

    relation_a.create_view("table_a")
    del relation_a

    relation_b = pt.duckdb.Relation("select 1 as b")
    relation_b.create_view("table_b")
    del relation_b

    default_database = pt.duckdb.Database.default()
    joined_relation = default_database.query(
        """
        select *
        from table_a
        inner join table_b
        on a = b
        """
    )
    assert joined_relation.to_df().frame_equal(pl.DataFrame({"a": [1], "b": [1]}))


def test_with_columns():
    """It should be able to crate new additional columns."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")

    # We can define a new column
    extended_relation = relation.with_columns(c="a + b")
    correct_extended = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    assert extended_relation.to_df().frame_equal(correct_extended)

    # Or even overwrite an existing column
    overwritten_relation = relation.with_columns(a="a + b")
    correct_overwritten = db.to_relation("select 2 as b, 3 as a").to_df()
    assert overwritten_relation.to_df().frame_equal(correct_overwritten)


def test_rename_to_existing_column():
    """Renaming a column to overwrite another should work."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")
    renamed_relation = relation.rename(b="a")
    assert renamed_relation.columns == ["a"]
    assert renamed_relation.get().a == 2


def test_add_suffix():
    """It should be able to add suffixes to all column names."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")
    assert relation.add_suffix("x").columns == ["ax", "bx"]
    assert relation.add_suffix("x", exclude=["a"]).columns == ["a", "bx"]
    assert relation.add_suffix("x", include=["a"]).columns == ["ax", "b"]

    with pytest.raises(
        TypeError,
        match="Both include and exclude provided at the same time!",
    ):
        relation.add_suffix("x", exclude=["a"], include=["b"])


def test_add_prefix():
    """It should be able to add prefixes to all column names."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")
    assert relation.add_prefix("x").columns == ["xa", "xb"]
    assert relation.add_prefix("x", exclude=["a"]).columns == ["a", "xb"]
    assert relation.add_prefix("x", include=["a"]).columns == ["xa", "b"]

    with pytest.raises(
        TypeError,
        match="Both include and exclude provided at the same time!",
    ):
        relation.add_prefix("x", exclude=["a"], include=["b"])


def test_relation_aggregate_method():
    """Test for Relation.aggregate()."""
    db = pt.duckdb.Database()
    relation = db.to_relation(
        pl.DataFrame(
            {
                "a": [1, 1, 2],
                "b": [10, 100, 1000],
                "c": [1, 2, 1],
            }
        )
    )
    aggregated_relation = relation.aggregate(
        "a",
        b_sum="sum(b)",
        group_by="a",
    )
    assert tuple(aggregated_relation) == (
        {"a": 1, "b_sum": 110},
        {"a": 2, "b_sum": 1000},
    )

    aggregated_relation_with_multiple_group_by = relation.aggregate(
        "a",
        "c",
        b_sum="sum(b)",
        group_by=["a", "c"],
    )
    assert tuple(aggregated_relation_with_multiple_group_by) == (
        {"a": 1, "c": 1, "b_sum": 10},
        {"a": 1, "c": 2, "b_sum": 100},
        {"a": 2, "c": 1, "b_sum": 1000},
    )


def test_relation_all_method():
    """Test for Relation.all()."""
    db = pt.duckdb.Database()
    relation = db.to_relation(
        pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [100, 100, 100],
            }
        )
    )

    assert not relation.all(a=100)
    assert relation.all(b=100)
    assert relation.all("a < 4", b=100)


def test_relation_case_method():
    db = pt.duckdb.Database()

    df = pl.DataFrame(
        {
            "shelf_classification": ["A", "B", "A", "C", "D"],
            "weight": [1, 2, 3, 4, 5],
        }
    )

    correct_df = df.with_columns(
        pl.Series([10, 20, 10, 0, None], dtype=pl.Int32).alias("max_weight")
    )
    correct_mapped_actions = db.to_relation(correct_df)

    mapped_actions = db.to_relation(df).case(
        from_column="shelf_classification",
        to_column="max_weight",
        mapping={"A": 10, "B": 20, "D": None},
        default=0,
    )
    assert mapped_actions == correct_mapped_actions

    # We can also use the Case class
    case_statement = pt.sql.Case(
        on_column="shelf_classification",
        mapping={"A": 10, "B": 20, "D": None},
        default=0,
    )
    alt_mapped_actions = db.to_relation(df).select(f"*, {case_statement} as max_weight")
    assert alt_mapped_actions == correct_mapped_actions


def test_relation_coalesce_method():
    """Test for Relation.coalesce()."""
    db = pt.duckdb.Database()
    df = pl.DataFrame(
        {"column_1": [1.0, None], "column_2": [None, "2"], "column_3": [3.0, None]}
    )
    relation = db.to_relation(df)
    coalesce_result = relation.coalesce(column_1=10, column_2="20").to_df()
    correct_coalesce_result = pl.DataFrame(
        {
            "column_1": [1.0, 10.0],
            "column_2": ["20", "2"],
            "column_3": [3.0, None],
        }
    )
    assert coalesce_result.frame_equal(correct_coalesce_result)


def test_relation_union_method():
    """Test for Relation.union and Relation.__add__."""
    db = pt.duckdb.Database()
    left = db.to_relation("select 1 as a, 2 as b")
    right = db.to_relation("select 200 as b, 100 as a")
    correct_union = pl.DataFrame(
        {
            "a": [1, 100],
            "b": [2, 200],
        }
    )
    assert left + right == correct_union
    assert right + left == correct_union[["b", "a"]][::-1]

    assert left.union(right) == correct_union
    assert right.union(left) == correct_union[["b", "a"]][::-1]

    incompatible = db.to_relation("select 1 as a")
    with pytest.raises(
        TypeError,
        match="Union between relations with different column names is not allowed.",
    ):
        incompatible + right  # pyright: ignore
    with pytest.raises(
        TypeError,
        match="Union between relations with different column names is not allowed.",
    ):
        left + incompatible  # pyright: ignore


def test_relation_model_functionality():
    """The end-user should be able to specify the constructor for row values."""
    db = pt.duckdb.Database()

    # We have two rows in our relation
    first_row_relation = db.to_relation("select 1 as a, 2 as b")
    second_row_relation = db.to_relation("select 3 as a, 4 as b")
    relation = first_row_relation + second_row_relation

    # Iterating over the relation should yield the same as .get()
    iterator_value = tuple(relation)[0]
    get_value = relation.get("a = 1")
    assert iterator_value == get_value
    assert iterator_value.a == 1
    assert get_value.a == 1
    assert iterator_value.b == 2
    assert get_value.b == 2

    # The end-user should be able to specify a custom row constructor
    model_mock = MagicMock(return_value="mock_return")
    new_relation = relation.set_model(model_mock)
    assert new_relation.get("a = 1") == "mock_return"
    model_mock.assert_called_with(a=1, b=2)

    # We create a custom model
    class MyModel(pt.Model):
        a: int
        b: str

    # Some dummy data
    dummy_df = MyModel.examples({"a": [1, 2], "b": ["one", "two"]})
    dummy_relation = db.to_relation(dummy_df)

    # Initially the relation has no custom model and it is dynamically constructed
    assert dummy_relation.model is None
    assert not isinstance(
        dummy_relation.limit(1).get(),
        MyModel,
    )

    # MyRow can be specified as the deserialization class with Relation.set_model()
    assert isinstance(
        dummy_relation.set_model(MyModel).limit(1).get(),
        MyModel,
    )

    # A custom relation class which specifies this as the default model
    class MyRelation(pt.duckdb.Relation):
        model = MyModel

    assert isinstance(
        MyRelation(dummy_relation._relation, database=db).limit(1).get(),
        MyModel,
    )

    # But the model is "lost" when we use schema-changing methods
    assert not isinstance(
        dummy_relation.set_model(MyModel).limit(1).select("a").get(),
        MyModel,
    )


def test_row_sql_type_functionality():
    """Tests for mapping pydantic types to DuckDB SQL types."""

    # Two nullable and two non-nullable columns
    class OptionalRow(pt.Model):
        a: str
        b: float
        c: Optional[str]
        d: Optional[bool]

    assert OptionalRow.non_nullable_columns == {"a", "b"}
    assert OptionalRow.nullable_columns == {"c", "d"}

    # All different types of SQL types
    class TypeModel(pt.Model):
        a: str
        b: int
        c: float
        d: Optional[bool]

    assert TypeModel.sql_types == {
        "a": "VARCHAR",
        "b": "INTEGER",
        "c": "DOUBLE",
        "d": "BOOLEAN",
    }


def test_fill_missing_columns():
    """Tests for Relation.with_missing_{nullable,defaultable}_columns."""

    class MyRow(pt.Model):
        # This can't be filled
        a: str
        # This can be filled with default value
        b: Optional[str] = "default_value"
        # This can be filled with null
        c: Optional[str]
        # This can be filled with null, but will be set
        d: Optional[float]
        # This can befilled with null, but with a different type
        e: Optional[bool]

    # We check if defaults are easily retrievable from the model
    assert MyRow.defaults == {"b": "default_value"}

    db = pt.duckdb.Database()
    df = pl.DataFrame({"a": ["mandatory"], "d": [10.5]})
    relation = db.to_relation(df).set_model(MyRow)

    # Missing nullable columns b, c, and e are filled in with nulls
    filled_nullables = relation.with_missing_nullable_columns()
    assert filled_nullables.set_model(None).get() == {
        "a": "mandatory",
        "b": None,
        "c": None,
        "d": 10.5,
        "e": None,
    }
    # And these nulls are properly typed
    assert filled_nullables.types == {
        "a": "VARCHAR",
        "b": "VARCHAR",
        "c": "VARCHAR",
        "d": "DOUBLE",
        "e": "BOOLEAN",
    }

    # Now we fill in the b column with "default_value"
    filled_defaults = relation.with_missing_defaultable_columns()
    assert filled_defaults.set_model(None).get().model_dump() == {
        "a": "mandatory",
        "b": "default_value",
        "d": 10.5,
    }
    assert filled_defaults.types == {
        "a": "VARCHAR",
        "b": "VARCHAR",
        "d": "DOUBLE",
    }

    # We now exclude the b column from being filled with default values
    excluded_default = relation.with_missing_defaultable_columns(exclude=["b"])
    assert excluded_default.set_model(None).get().model_dump() == {
        "a": "mandatory",
        "d": 10.5,
    }

    # We can also specify that we only want to fill a subset
    included_defualts = relation.with_missing_defaultable_columns(include=["b"])
    assert included_defualts.set_model(None).get().model_dump() == {
        "a": "mandatory",
        "b": "default_value",
        "d": 10.5,
    }

    # We now exclude column b and c from being filled with null values
    excluded_nulls = relation.with_missing_nullable_columns(exclude=["b", "c"])
    assert excluded_nulls.set_model(None).get().model_dump() == {
        "a": "mandatory",
        "d": 10.5,
        "e": None,
    }

    # Only specify that we want to fill column e with nulls
    included_nulls = relation.with_missing_nullable_columns(include=["e"])
    assert included_nulls.set_model(None).get().model_dump() == {
        "a": "mandatory",
        "d": 10.5,
        "e": None,
    }

    # We should raise if both include and exclude is specified
    with pytest.raises(
        TypeError, match="Both include and exclude provided at the same time!"
    ):
        relation.with_missing_nullable_columns(include={"x"}, exclude={"y"})

    with pytest.raises(
        TypeError, match="Both include and exclude provided at the same time!"
    ):
        relation.with_missing_defaultable_columns(include={"x"}, exclude={"y"})


def test_with_missing_nullable_enum_columns():
    """It should produce enums with null values correctly."""

    class EnumModel(pt.Model):
        enum_column: Optional[Literal["a", "b", "c"]]
        other_column: int

    db = pt.duckdb.Database()

    # We insert data into a properly typed table in order to get the correct enum type
    db.create_table(name="enum_table", model=EnumModel)
    db.to_relation("select 'a' as enum_column, 1 as other_column").insert_into(
        table="enum_table"
    )
    table_relation = db.table("enum_table")
    assert str(table_relation.types["enum_column"]).startswith("enum__")

    # We generate another dynamic relation where we expect the correct enum type
    null_relation = (
        db.to_relation("select 2 as other_column")
        .set_model(EnumModel)
        .with_missing_nullable_columns()
    )
    assert null_relation.types["enum_column"] == table_relation.types["enum_column"]

    # These two relations should now be unionable
    union_relation = (null_relation + table_relation).order("other_column asc")
    assert union_relation.types["enum_column"] == table_relation.types["enum_column"]

    with pl.StringCache():
        correct_union_df = pl.DataFrame(
            {
                "other_column": [1, 2],
                "enum_column": pl.Series(["a", None]).cast(pl.Categorical),
            }
        )
        assert union_relation.to_df().frame_equal(correct_union_df)


def test_with_missing_nullable_enum_columns_without_table():
    """It should produce enums with null values correctly without a table."""

    class EnumModel(pt.Model):
        enum_column_1: Optional[Literal["a", "b", "c"]]
        enum_column_2: Optional[Literal["a", "b", "c"]]
        other_column: int

    # We should be able to create the correct type without a table
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as other_column")
    with pytest.raises(
        TypeError, match=r".*You should invoke Relation.set_model\(\) first!"
    ):
        relation.with_missing_nullable_columns()

    model_relation = relation.set_model(EnumModel).with_missing_nullable_columns()
    assert str(model_relation.types["enum_column_1"]).startswith("enum__")
    assert (
        model_relation.types["enum_column_2"] == model_relation.types["enum_column_1"]
    )

    # And now we should be able to insert it into a new table
    model_relation.create_table(name="enum_table")
    table_relation = db.table("enum_table")
    assert (
        table_relation.types["enum_column_1"] == model_relation.types["enum_column_1"]
    )
    assert (
        table_relation.types["enum_column_2"] == model_relation.types["enum_column_1"]
    )


def test_with_missing_defualtable_enum_columns():
    """It should produce enums with default values correctly typed."""

    class EnumModel(pt.Model):
        enum_column: Optional[Literal["a", "b", "c"]] = "a"
        other_column: int

    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as other_column")
    with pytest.raises(
        TypeError,
        match=r".*You should invoke Relation.set_model\(\) first!",
    ):
        relation.with_missing_defaultable_columns()

    model_relation = relation.set_model(EnumModel).with_missing_defaultable_columns()
    assert str(model_relation.types["enum_column"]).startswith("enum__")


def test_relation_insert_into():
    """Relation.insert_into() should automatically order columnns correctly."""
    db = pt.duckdb.Database()
    db.execute(
        """
        create table foo (
            a integer,
            b integer
        )
    """
    )
    db.to_relation("select 2 as b, 1 as a").insert_into(table="foo")
    row = db.table("foo").get()
    assert row.a == 1
    assert row.b == 2

    with pytest.raises(
        TypeError,
        match=re.escape(
            "Relation is missing column(s) {'a'} "
            "in order to be inserted into table 'foo'!"
        ),
    ):
        db.to_relation("select 2 as b, 1 as c").insert_into(table="foo")


def test_polars_support():
    # Test converting a polars DataFrame to patito relation
    df = pl.DataFrame(data={"column_1": ["a", "b", None], "column_2": [1, 2, None]})
    correct_dtypes = [pl.Utf8, pl.Int64]
    assert df.dtypes == correct_dtypes
    db = pt.duckdb.Database()
    relation = db.to_relation(df)
    assert relation.get(column_1="a").column_2 == 1

    # Test converting back again the other way
    roundtrip_df = relation.to_df()
    assert roundtrip_df.frame_equal(df)
    assert roundtrip_df.dtypes == correct_dtypes

    # Assert that .to_df() always returns a DataFrame.
    assert isinstance(relation["column_1"].to_df(), pl.DataFrame)

    # Assert that .to_df() returns an empty DataFrame when the table has no rows
    empty_dataframe = relation.filter(column_1="missing-column").to_df()
    # assert empty_dataframe == pl.DataFrame(columns=["column_1", "column_2"])
    # assert empty_dataframe.frame_equal(pl.DataFrame(columns=["column_1", "column_2"]))

    # The datatype should be preserved
    assert empty_dataframe.dtypes == correct_dtypes

    # A model should be able to be instantiated with a polars row
    class MyModel(pt.Model):
        a: int
        b: str

    my_model_df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    with pytest.raises(
        ValueError,
        match=r"MyModel._from_polars\(\) can only be invoked with exactly 1 row.*",
    ):
        MyModel.from_row(my_model_df)

    my_model = MyModel.from_row(my_model_df.head(1))
    assert my_model.a == 1
    assert my_model.b == "x"

    # Anything besides a polars dataframe should raise TypeError
    with pytest.raises(TypeError):
        MyModel.from_row(None)  # pyright: ignore

    # But we can also skip validation if we want
    unvalidated_model = MyModel.from_row(
        pl.DataFrame().with_columns(
            [
                pl.lit("string").alias("a"),
                pl.lit(2).alias("b"),
            ]
        ),
        validate=False,
    )
    assert unvalidated_model.a == "string"
    assert unvalidated_model.b == 2


def test_series_vs_dataframe_behavior():
    """Test Relation.to_series()."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as column_1, 2 as column_2")

    # Selecting multiple columns should yield a DataFrame
    assert isinstance(relation[["column_1", "column_2"]].to_df(), pl.DataFrame)

    # Selecting a single column, but as an item in a list, should yield a DataFrame
    assert isinstance(relation[["column_1"]].to_df(), pl.DataFrame)

    # Selecting a single column as a string should also yield a DataFrame
    assert isinstance(relation["column_1"].to_df(), pl.DataFrame)

    # But .to_series() should yield a series
    series = relation["column_1"].to_series()
    assert isinstance(series, pl.Series)

    # The name should also be set correctly
    assert series.name == "column_1"

    # And the content should be correct
    correct_series = pl.Series([1], dtype=pl.Int32).alias("column_1")
    assert series.series_equal(correct_series)

    # To series will raise a type error if invoked with anything other than 1 column
    with pytest.raises(TypeError, match=r".*2 columns, while exactly 1 is required.*"):
        relation.to_series()


def test_converting_enum_column_to_polars():
    """Enum types should be convertible to polars categoricals."""

    class EnumModel(pt.Model):
        enum_column: Literal["a", "b", "c"]

    db = pt.duckdb.Database()
    db.create_table(name="enum_table", model=EnumModel)
    db.execute(
        """
        insert into enum_table
            (enum_column)
        values
            ('a'),
            ('a'),
            ('b');
        """
    )
    enum_df = db.table("enum_table").to_df()
    assert enum_df.frame_equal(pl.DataFrame({"enum_column": ["a", "a", "b"]}))
    assert enum_df.dtypes == [pl.Categorical]


def test_non_string_enum():
    """It should handle other types than just string enums."""

    class EnumModel(pt.Model):
        enum_column: Literal[10, 11, 12]

    db = pt.duckdb.Database()
    db.create_table(name="enum_table", model=EnumModel)

    db.execute(
        """
        insert into enum_table
            (enum_column)
        values
            (10),
            (11),
            (12);
        """
    )
    enum_df = db.table("enum_table").to_df()
    assert enum_df.frame_equal(pl.DataFrame({"enum_column": [10, 11, 12]}))
    assert enum_df.dtypes == [pl.Int64]


def test_multiple_filters():
    """The filter method should AND multiple filters properly."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")
    # The logical or should not make the filter valid for our row
    assert relation.filter("(1 = 2) or b = 2", a=0).count() == 0
    assert relation.filter("a=0", "(1 = 2) or b = 2").count() == 0


def test_no_filter():
    """No filters should return all rows."""
    db = pt.duckdb.Database()
    relation = db.to_relation("select 1 as a, 2 as b")
    # The logical or should not make the filter valid for our row
    assert relation.filter().count()


def test_string_representation_of_relation():
    """It should have a string representation."""
    relation = pt.duckdb.Relation("select 1 as my_column")
    relation_str = str(relation)
    assert "my_column" in relation_str


def test_cast():
    """It should be able to cast to the correct SQL types based on model."""

    class Schema(pt.Model):
        float_column: float

    relation = pt.duckdb.Relation("select 1 as float_column, 2 as other_column")
    with pytest.raises(
        TypeError,
        match=(
            r"Relation\.cast\(\) invoked without Relation.model having been set\! "
            r"You should invoke Relation\.set_model\(\) first or explicitly provide "
            r"a model to \.cast\(\)."
        ),
    ):
        relation.cast()

    # Originally the type of both columns are integers
    modeled_relation = relation.set_model(Schema)
    assert modeled_relation.types["float_column"] == "INTEGER"
    assert modeled_relation.types["other_column"] == "INTEGER"

    # The casted variant has converted the float column to double
    casted_relation = relation.set_model(Schema).cast()
    assert casted_relation.types["float_column"] == "DOUBLE"
    # But kept the other as-is
    assert casted_relation.types["other_column"] == "INTEGER"

    # You can either set the model with .set_model() or provide it to cast
    assert (
        relation.set_model(Schema)
        .cast()
        .to_df()
        .frame_equal(relation.cast(Schema).to_df())
    )

    # Other types that should be considered compatible should be kept as-is
    compatible_relation = pt.duckdb.Relation("select 1::FLOAT as float_column")
    assert compatible_relation.cast(Schema).types["float_column"] == "FLOAT"

    # Unless the strict parameter is specified
    assert (
        compatible_relation.cast(Schema, strict=True).types["float_column"] == "DOUBLE"
    )

    # We can also specify a specific SQL type
    class SpecificSQLTypeSchema(pt.Model):
        float_column: float = pt.Field(sql_type="BIGINT")

    specific_cast_relation = relation.set_model(SpecificSQLTypeSchema).cast()
    assert specific_cast_relation.types["float_column"] == "BIGINT"

    # Unknown types raise
    class ObjectModel(pt.Model):
        object_column: object

    with pytest.raises(
        NotImplementedError,
        match=r"No valid sql_type mapping found for column 'object_column'\.",
    ):
        pt.duckdb.Relation("select 1 as object_column").set_model(ObjectModel).cast()

    # Check for more specific type annotations
    class TotalModel(pt.Model):
        timedelta_column: timedelta
        date_column: date
        null_column: None

    df = pt.DataFrame(
        {
            "date_column": [date(2022, 9, 4)],
            "null_column": [None],
        }
    )
    casted_relation = pt.duckdb.Relation(df, model=TotalModel).cast()
    assert casted_relation.types == {
        "date_column": "DATE",
        "null_column": "INTEGER",
    }
    assert casted_relation.to_df().frame_equal(df)

    # It is possible to only cast a subset
    class MyModel(pt.Model):
        column_1: float
        column_2: float

    relation = pt.duckdb.Relation("select 1 as column_1, 2 as column_2").set_model(
        MyModel
    )
    assert relation.cast(include=[]).types == {
        "column_1": "INTEGER",
        "column_2": "INTEGER",
    }
    assert relation.cast(include=["column_1"]).types == {
        "column_1": "DOUBLE",
        "column_2": "INTEGER",
    }
    assert relation.cast(include=["column_1", "column_2"]).types == {
        "column_1": "DOUBLE",
        "column_2": "DOUBLE",
    }

    assert relation.cast(exclude=[]).types == {
        "column_1": "DOUBLE",
        "column_2": "DOUBLE",
    }
    assert relation.cast(exclude=["column_1"]).types == {
        "column_1": "INTEGER",
        "column_2": "DOUBLE",
    }
    assert relation.cast(exclude=["column_1", "column_2"]).types == {
        "column_1": "INTEGER",
        "column_2": "INTEGER",
    }

    # Providing both include and exclude should raise a value error
    with pytest.raises(
        ValueError,
        match=r"Both include and exclude provided to Relation.cast\(\)\!",
    ):
        relation.cast(include=["column_1"], exclude=["column_2"])


@pytest.mark.xfail(strict=True)
def test_casting_timedelta_column_back_and_forth():
    class TotalModel(pt.Model):
        timedelta_column: timedelta
        date_column: date
        null_column: None

    df = pt.DataFrame(
        {
            "timedelta_column": [timedelta(seconds=90)],
            "date_column": [date(2022, 9, 4)],
            "null_column": [None],
        }
    )
    casted_relation = pt.duckdb.Relation(df, model=TotalModel).cast()
    assert casted_relation.types == {
        "timedelta_column": "INTERVAL",
        "date_column": "DATE",
        "null_column": "INTEGER",
    }
    assert casted_relation.to_df().frame_equal(df)
