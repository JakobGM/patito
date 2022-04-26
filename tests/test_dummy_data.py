"""Test of functionality related to the generation of dummy data."""
from datetime import date, datetime
from typing import Optional

import polars as pl
import pytest

import patito as pt


def test_model_example_df():
    """Test for patito.Model.example()."""

    # When inheriting from Model you get a .example() method for generating dataframes
    # default values according to the type annotation.
    class MyRow(pt.Model):
        a: int
        b: int
        c: int
        d: int
        e: str

    df_1 = MyRow.example({"a": [1, 2], "b": [3, 4], "c": 5})
    df_2 = MyRow.example(
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
        MyRow.example({"a": [0], "f": [1], "g": [2]})


def test_example():
    class MyModel(pt.Model):
        a: int
        b: Optional[str]
        c: Optional[int]

    df = MyModel.example({"a": [1, 2]})
    assert isinstance(df, pl.DataFrame)
    assert df.dtypes == [pl.Int64, pl.Utf8, pl.Int64]
    assert df.columns == ["a", "b", "c"]


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

    example_df = UniqueModel.example({"bool_column": [True, False]})
    for column in UniqueModel.columns:
        assert example_df[column].is_duplicated().sum() == 0
