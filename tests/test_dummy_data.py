"""Test of functionality related to the generation of dummy data."""
from datetime import date, datetime
from typing import Optional

import polars as pl
import pytest
from typing_extensions import Literal

import patito as pt


def test_model_example_df():
    """Test for patito.Model.example()."""

    # When inheriting from Model you get a .examples() method for generating dataframes
    # default values according to the type annotation.
    class MyRow(pt.Model):
        a: int
        b: int
        c: int
        d: int
        e: str

    df_1 = MyRow.examples({"a": [1, 2], "b": [3, 4], "c": 5})
    df_2 = MyRow.examples(
        data=[[1, 3, 5], [2, 4, 5]],
        columns=["a", "b", "c"],
    )
    correct_df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 5],
            "d": [-1, -1],
            "e": ["dummy_string", "dummy_string"],
        }
    )

    assert df_1[correct_df.columns].frame_equal(correct_df)
    assert df_2[correct_df.columns].frame_equal(correct_df)

    # A TypeError should be raised when you provide wrong keywords
    with pytest.raises(
        TypeError,
        match="MyRow does not contain fields {'[fg]', '[fg]'}!",
    ):
        MyRow.examples({"a": [0], "f": [1], "g": [2]})


def test_examples():
    class MyModel(pt.Model):
        a: int
        b: Optional[str]
        c: Optional[int]

    df = MyModel.examples({"a": [1, 2]})
    assert isinstance(df, pl.DataFrame)
    assert df.dtypes == [pl.Int64, pl.Utf8, pl.Int64]
    assert df.columns == ["a", "b", "c"]

    # A TypeError should be raised when you provide no column names
    with pytest.raises(
        TypeError,
        match=r"MyModel\.examples\(\) must be provided with column names\!",
    ):
        MyModel.examples([[1, 2]])


@pytest.mark.skipif("Database" not in dir(pt), reason="Requires DuckDB")
def test_creation_of_empty_relation():
    """You should be able to create a zero-row relation with correct types."""

    class MyModel(pt.Model):
        a: int
        b: Optional[str]

    db = pt.Database()
    empty_relation = db.empty_relation(schema=MyModel)
    assert empty_relation.columns == ["a", "b"]
    assert empty_relation.types == ["BIGINT", "VARCHAR"]
    assert empty_relation.count() == 0


def test_generation_of_unique_data():
    """Example data generators should be able to generate unique data."""

    class UniqueModel(pt.Model):
        bool_column: bool
        string_column: str = pt.Field(unique=True)
        int_column: int = pt.Field(unique=True)
        float_column: int = pt.Field(unique=True)
        date_column: date = pt.Field(unique=True)
        datetime_column: datetime = pt.Field(unique=True)

    example_df = UniqueModel.examples({"bool_column": [True, False]})
    for column in UniqueModel.columns:
        assert example_df[column].is_duplicated().sum() == 0


def test_enum_field_example_values():
    """It should produce correct example values for enums."""

    class DefaultEnumModel(pt.Model):
        row_number: int
        # Here the first value will be used as the example value
        enum_field: Literal["a", "b", "c"]
        # Here the default value will be used as the example value
        default_enum_field: Literal["a", "b", "c"] = "b"
        default_optional_enum_field: Optional[Literal["a", "b", "c"]] = "c"
        # Here null will be used as the example value
        none_default_optional_enum_field: Optional[Literal["a", "b", "c"]] = None

    example_df = DefaultEnumModel.examples({"row_number": [1]})
    correct_example_df = pl.DataFrame(
        [
            pl.Series("row_number", [1], dtype=pl.Int64),
            pl.Series("enum_field", ["a"], dtype=pl.Categorical),
            pl.Series("default_enum_field", ["b"], dtype=pl.Categorical),
            pl.Series("default_optional_enum_field", ["c"], dtype=pl.Categorical),
            pl.Series("none_default_optional_enum_field", [None], dtype=pl.Categorical),
        ]
    )

    # Workaround for pola-rs/polars#4253
    assert example_df.with_column(
        pl.col("none_default_optional_enum_field").cast(pl.Categorical)
    ).frame_equal(correct_example_df)

    example_model = DefaultEnumModel.example()
    assert example_model.enum_field == "a"
    assert example_model.default_enum_field == "b"
    assert example_model.default_optional_enum_field == "c"
    assert example_model.none_default_optional_enum_field is None
