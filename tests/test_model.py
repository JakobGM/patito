"""Tests for patito.Model."""
# pyright: reportPrivateImportUsage=false
from datetime import date, datetime, timedelta
from typing import Optional

import polars as pl
import pytest
from typing_extensions import Literal

import patito as pt


def test_model_example():
    """Test for Model.example()."""

    # When inheriting from Model you get a .dummy() method for generating rows with
    # default values according to the type annotation.
    class MyModel(pt.Model):
        int_value: int
        float_value: float
        str_value: str
        bool_value: bool
        literal_value: Literal["a", "b"]
        default_value: str = "my_default"
        optional_value: Optional[int]
        bounded_value: int = pt.Field(ge=10, le=20)
        date_value: date
        datetime_value: datetime

    assert MyModel.example().dict() == {
        "int_value": -1,
        "float_value": -0.5,
        "str_value": "dummy_string",
        "bool_value": False,
        "literal_value": "a",
        "default_value": "my_default",
        "optional_value": None,
        "bounded_value": 15,
        "date_value": date(year=1970, month=1, day=1),
        "datetime_value": datetime(year=1970, month=1, day=1),
    }
    assert MyModel.example(
        bool_value=True,
        default_value="override",
        optional_value=1,
    ).dict() == {
        "int_value": -1,
        "float_value": -0.5,
        "str_value": "dummy_string",
        "bool_value": True,
        "literal_value": "a",
        "default_value": "override",
        "optional_value": 1,
        "bounded_value": 15,
        "date_value": date(year=1970, month=1, day=1),
        "datetime_value": datetime(year=1970, month=1, day=1),
    }

    # For now, valid regex data is not implemented
    class RegexModel(pt.Model):
        regex_column: str = pt.Field(regex=r"[0-9a-f]")

    with pytest.raises(
        NotImplementedError,
        match="Example data generation has not been implemented for regex.*",
    ):
        RegexModel.example()


def test_model_pandas_examples():
    """Test for Row.dummy_pandas()."""
    pd = pytest.importorskip("pandas")

    # When inheriting from Model you get a .dummy_df() method for generating dataframes
    # default values according to the type annotation.
    class MyRow(pt.Model):
        a: int
        b: int
        c: int
        d: int
        e: str

    df_1 = MyRow.pandas_examples({"a": [1, 2], "b": [3, 4], "c": 5})
    df_2 = MyRow.pandas_examples(
        data=[[1, 3, 5], [2, 4, 5]],
        columns=["a", "b", "c"],
    )
    correct_df = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "c": [5, 5],
            "d": [-1, -1],
            "e": ["dummy_string", "dummy_string"],
        }
    )

    pd.testing.assert_frame_equal(df_1, correct_df)
    pd.testing.assert_frame_equal(df_2, correct_df)

    # A TypeError should be raised when you provide wrong keywords
    with pytest.raises(
        TypeError,
        match="MyRow does not contain fields {'[fg]', '[fg]'}!",
    ):
        MyRow.pandas_examples({"a": [0], "f": [1], "g": [2]})

    # A TypeError should be raised when you provide no column names
    with pytest.raises(
        TypeError,
        match=r"MyRow\.pandas_examples\(\) must be provided with column names\!",
    ):
        MyRow.pandas_examples([[1, 2, 3, 4]])


def test_instantiating_model_from_row():
    """You should be able to instantiate models from rows."""

    class Model(pt.Model):
        a: int

    polars_dataframe = pl.DataFrame({"a": [1]})
    assert Model.from_row(polars_dataframe).a == 1

    # Anything besides a dataframe / row should raise TypeError
    with pytest.raises(
        TypeError, match=r"Model.from_row not implemented for \<class 'NoneType'\>."
    ):
        Model.from_row(None)  # type: ignore


def test_insstantiation_from_pandas_row():
    """You should be able to instantiate models from pandas rows."""
    pytest.importorskip("pandas")

    class Model(pt.Model):
        a: int

    polars_dataframe = pl.DataFrame({"a": [1]})
    assert Model.from_row(polars_dataframe).a == 1

    pandas_dataframe = polars_dataframe.to_pandas()
    assert Model.from_row(pandas_dataframe).a == 1
    assert Model.from_row(pandas_dataframe.loc[0]).a == 1


def test_model_dataframe_class_creation():
    """Each model should get a custom DataFrame class."""

    class CustomModel(pt.Model):
        a: int

    # The DataFrame class is a sub-class of patito.DataFrame
    assert issubclass(CustomModel.DataFrame, pt.DataFrame)

    # And the model
    assert CustomModel.DataFrame.model is CustomModel


def test_mapping_to_polars_dtypes():
    """Model fields should be mappable to polars dtypes."""

    class CompleteModel(pt.Model):
        str_column: str
        int_column: int
        float_column: float
        bool_column: bool

        date_column: date
        datetime_column: datetime
        duration_column: timedelta

        categorical_column: Literal["a", "b", "c"]
        null_column: None

    assert CompleteModel.dtypes == {
        "str_column": pl.Utf8,
        "int_column": pl.Int64,
        "float_column": pl.Float64,
        "bool_column": pl.Boolean,
        "date_column": pl.Date,
        "datetime_column": pl.Datetime,
        "duration_column": pl.Duration,
        "categorical_column": pl.Categorical,
        "null_column": pl.Null,
    }

    assert CompleteModel.valid_dtypes == {
        "str_column": [pl.Utf8],
        "int_column": [
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
        ],
        "float_column": [pl.Float64, pl.Float32],
        "bool_column": [pl.Boolean],
        "date_column": [pl.Date],
        "datetime_column": [pl.Datetime],
        "duration_column": [pl.Duration],
        "categorical_column": [pl.Categorical, pl.Utf8],
        "null_column": [pl.Null],
    }
