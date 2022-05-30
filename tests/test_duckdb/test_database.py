"""Tests for patito.Database."""
from typing import Optional

import polars as pl
import pytest
from typing_extensions import Literal

import patito as pt

# Skip test module if DuckDB is not installed
if not pt._DUCKDB_AVAILABLE:
    pytest.skip("DuckDB not installed", allow_module_level=True)


def test_database(tmp_path):
    """Test functionality of Database class."""
    # Create a new in-memory database
    db = pt.Database()

    # Insert a simple dataframe as a new table
    table_df = pl.DataFrame(
        {
            "column_1": [1, 2, 3],
            "column_2": ["a", "b", "c"],
        }
    )
    db.to_relation(table_df).create_table(name="table_name_1")

    # Check that a round-trip to and from the database preserves the data
    db_table = db.table("table_name_1").to_df()
    assert db_table is not table_df
    assert table_df.frame_equal(db_table)
    pl.testing.assert_frame_equal(table_df, db_table)

    # Check that new database objects are isolated from previous ones
    another_db = pt.Database()
    with pytest.raises(RuntimeError, match="Table does not exist!"):
        db_table = another_db.table("table_name_1")

    # Check the parquet reading functionality
    parquet_path = tmp_path / "tmp.parquet"
    table_df.write_parquet(str(parquet_path), compression="snappy")
    new_relation = another_db.to_relation(parquet_path)
    new_relation.create_table(name="parquet_table")
    assert another_db.table("parquet_table").count() == 3


def test_file_database(tmp_path):
    """Check if the Database can be persisted to a file."""
    # Insert some data into a file-backed database
    db_path = tmp_path / "tmp.db"
    file_db = pt.Database(path=db_path)
    file_db.to_relation("select 1 as a, 2 as b").create_table(name="table")
    before_df = file_db.table("table").to_df()

    # Delete the database
    del file_db

    # And restore tha data from the file
    restored_db = pt.Database(path=db_path)
    after_df = restored_db.table("table").to_df()

    # The data should still be the same
    assert before_df.frame_equal(after_df)


def test_database_create_table():
    """Tests for patito.Database.create_table()."""

    # A pydantic basemodel is used to specify the table schema
    # We inherit here in order to make sure that inheritance works as intended
    class BaseModel(pt.Model):
        int_column: int
        optional_int_column: Optional[int]
        str_column: str

    class Model(BaseModel):
        optional_str_column: Optional[str]
        bool_column: bool
        optional_bool_column: Optional[bool]
        enum_column: Literal["a", "b", "c"]

    # We crate the table schema
    db = pt.Database()
    table = db.create_table(name="test_table", model=Model)

    # We insert some dummy data into the new table
    dummy_relation = db.to_relation(Model.examples({"optional_int_column": [1, None]}))
    dummy_relation.insert_into(table_name="test_table")

    # But we should not be able to insert null data in non-optional columns
    null_relation = dummy_relation.drop("int_column").project("null as int_column, *")
    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to insert into table 'test_table': Constraint Error:.*"
            "NOT NULL constraint failed: test_table.int_column"
        ),
    ):
        null_relation.insert_into(table_name="test_table")

    # Check if the correct columns and types have been set
    assert table.columns == [
        "int_column",
        "optional_int_column",
        "str_column",
        "optional_str_column",
        "bool_column",
        "optional_bool_column",
        "enum_column",
    ]
    assert table.types == [
        "BIGINT",
        "BIGINT",
        "VARCHAR",
        "VARCHAR",
        "BOOLEAN",
        "BOOLEAN",
        "model__enum_column",
    ]


def test_validate_non_nullabel_enum_columns():
    """Enum columns should be null-validated."""

    class EnumModel(pt.Model):
        non_nullable_enum_column: Literal["a", "b", "c"]
        nullable_enum_column: Optional[Literal["a", "b", "c"]]

    db = pt.Database()
    db.create_table(name="enum_table", model=EnumModel)

    # We allow null values in nullable_enum_column
    valid_relation = db.to_relation(
        "select 'a' as non_nullable_enum_column, null as nullable_enum_column"
    )
    valid_relation.insert_into("enum_table")

    # But we do not allow it in non_nullable_enum_column
    invalid_relation = db.to_relation(
        "select null as non_nullable_enum_column, 'a' as nullable_enum_column"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to insert into table 'enum_table': Constraint Error:.*"
            "NOT NULL constraint failed: enum_table.non_nullable_enum_column"
        ),
    ):
        invalid_relation.insert_into(table_name="enum_table")

    # The non-nullable enum column should do enum value validation
    invalid_relation = db.to_relation(
        "select 'd' as non_nullable_enum_column, 'a' as nullable_enum_column"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to insert into table 'enum_table': Conversion Error:.*"
            "Could not convert string 'd' to UINT8"
        ),
    ):
        invalid_relation.insert_into(table_name="enum_table")

    # And the nullable enum column should do enum value validation
    invalid_relation = db.to_relation(
        "select 'a' as non_nullable_enum_column, 'd' as nullable_enum_column"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to insert into table 'enum_table': Conversion Error:.*"
            "Could not convert string 'd' to UINT8"
        ),
    ):
        invalid_relation.insert_into(table_name="enum_table")


def test_table_existence_check():
    """You should be able to check for the existence of a table."""

    class Model(pt.Model):
        column_1: str
        column_2: int

    # At first there is no table named "test_table"
    db = pt.Database()
    assert "test_table" not in db

    # We create the table
    db.create_table(name="test_table", model=Model)

    # And now the table should exist
    assert "test_table" in db
