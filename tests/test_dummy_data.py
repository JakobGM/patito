"""Test of functionality related to the generation of dummy data."""

from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, Optional

import polars as pl
import pytest

import patito as pt


def test_model_example_df() -> None:
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

    assert df_1[correct_df.columns].equals(correct_df)
    assert df_2[correct_df.columns].equals(correct_df)

    # A TypeError should be raised when you provide wrong keywords
    with pytest.raises(
        TypeError,
        match="MyRow does not contain fields {'[fg]', '[fg]'}!",
    ):
        MyRow.examples({"a": [0], "f": [1], "g": [2]})


def test_examples() -> None:
    """Test model.examples()."""

    class MyModel(pt.Model):
        a: int
        b: Optional[str]
        c: Optional[int]
        d: Optional[list[str]] = pt.Field(dtype=pl.List(pl.String))
        e: list[int]
        f: int = pt.Field(ge=0)

    df = MyModel.examples({"a": [1, 2]})
    assert isinstance(df, pl.DataFrame)
    assert df.dtypes == [
        pl.Int64,
        pl.String,
        pl.Int64,
        pl.List(pl.String),
        pl.List(pl.Int64),
        pl.Int64,
    ]
    assert df.columns == ["a", "b", "c", "d", "e", "f"]
    assert (df["f"] >= 0).all()
    MyModel.validate(df)

    # A TypeError should be raised when you provide no column names
    with pytest.raises(
        TypeError,
        match=r"MyModel\.examples\(\) must be provided with column names\!",
    ):
        MyModel.examples([[1, 2]])


def test_generation_of_unique_data() -> None:
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


def test_enum_field_example_values() -> None:
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

    # Workaround for pola-rs/polars#4253
    example_df = DefaultEnumModel.examples({"row_number": [1]}).with_columns(
        pl.col("none_default_optional_enum_field").cast(pl.Enum(["a", "b", "c"]))
    )

    correct_example_df = pl.DataFrame(
        [
            pl.Series("row_number", [1], dtype=pl.Int64),
            pl.Series("enum_field", ["a"], dtype=pl.Enum(["a", "b", "c"])),
            pl.Series("default_enum_field", ["b"], dtype=pl.Enum(["a", "b", "c"])),
            pl.Series(
                "default_optional_enum_field", ["c"], dtype=pl.Enum(["a", "b", "c"])
            ),
            pl.Series(
                "none_default_optional_enum_field",
                [None],
                dtype=pl.Enum(["a", "b", "c"]),
            ),
        ]
    )

    # Workaround for pl.StringCache() not working here for some reason
    assert correct_example_df.dtypes == example_df.dtypes
    assert example_df.select(pl.all().cast(pl.String)).equals(
        correct_example_df.select(pl.all().cast(pl.String))
    )

    example_model = DefaultEnumModel.example()
    assert example_model.enum_field == "a"
    assert example_model.default_enum_field == "b"
    assert example_model.default_optional_enum_field == "c"
    assert example_model.none_default_optional_enum_field is None


def test_nested_models() -> None:
    """It should be possible to create nested models."""

    class NestedModel(pt.Model):
        nested_field: int

    class ParentModel1(pt.Model):
        parent_field: int
        nested_model: NestedModel

    example_model = ParentModel1.example()
    example_df = ParentModel1.examples()
    assert isinstance(example_model.nested_model, NestedModel)
    assert example_model.nested_model.nested_field is not None

    example_df = ParentModel1.examples()
    assert isinstance(example_df, pl.DataFrame)

    # inheritance also works
    class ParentModel2(NestedModel):
        parent_field: int

    example_model = ParentModel2.example()
    assert example_model.nested_field is not None
    assert example_model.parent_field is not None

    # and optional nested models are ok
    class ParentModel3(pt.Model):
        parent_field: int
        nested_model: Optional[NestedModel] = None

    example_model = ParentModel3.example()
    assert example_model.nested_model is None

    # sequences of nested models also work
    class ParentModel(pt.Model):
        parent_field: int
        nested_models: Sequence[NestedModel]

    example_model = ParentModel.example()
    ParentModel.examples()
