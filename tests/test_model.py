"""Tests for patito.Model."""
# pyright: reportPrivateImportUsage=false
import enum
import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Type, Literal

import polars as pl
import pytest
from pydantic import ValidationError

import patito as pt
from patito.pydantic import PL_INTEGER_DTYPES


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

    assert MyModel.example().model_dump() == {
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
    ).model_dump() == {
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
        regex_column: str = pt.Field(pattern=r"[0-9a-f]")

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
        Model.from_row(None)  # pyright: ignore

    with pytest.raises(
        TypeError,
        match=r"Model._from_polars\(\) must be invoked with polars.DataFrame.*",
    ):
        Model._from_polars(None)  # pyright: ignore


def test_insstantiation_from_pandas_row():
    """You should be able to instantiate models from pandas rows."""
    pytest.importorskip("pandas")

    class Model(pt.Model):
        a: int

    polars_dataframe = pl.DataFrame({"a": [1]})
    assert Model.from_row(polars_dataframe).a == 1

    pandas_dataframe = polars_dataframe.to_pandas()
    assert Model.from_row(pandas_dataframe).a == 1
    assert Model.from_row(pandas_dataframe.loc[0]).a == 1  # type: ignore


def test_model_dataframe_class_creation():
    """Each model should get a custom DataFrame class."""

    class CustomModel(pt.Model):
        a: int

    # The DataFrame class is a sub-class of patito.DataFrame
    assert issubclass(CustomModel.DataFrame, pt.DataFrame)

    # The LazyFrame class is a sub-class of patito.LazyFrame
    assert issubclass(CustomModel.LazyFrame, pt.LazyFrame)

    # And the model
    assert CustomModel.DataFrame.model is CustomModel
    assert CustomModel.LazyFrame.model is CustomModel


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


def test_mapping_to_polars_dtypes_with_lists():
    """Model list fields should be mappable to polars dtypes."""

    class CompleteModel(pt.Model):
        str_column: List[str]
        int_column: List[int]
        float_column: List[float]
        bool_column: List[bool]

        date_column: List[date]
        datetime_column: List[datetime]
        duration_column: List[timedelta]

        categorical_column: List[Literal["a", "b", "c"]]
        null_column: List[None]

    assert CompleteModel.dtypes == {
        "str_column": pl.List(pl.Utf8),
        "int_column": pl.List(pl.Int64),
        "float_column": pl.List(pl.Float64),
        "bool_column": pl.List(pl.Boolean),
        "date_column": pl.List(pl.Date),
        "datetime_column": pl.List(pl.Datetime),
        "duration_column": pl.List(pl.Duration),
        "categorical_column": pl.List(pl.Categorical),
        "null_column": pl.List(pl.Null),
    }

    assert CompleteModel.valid_dtypes == {
        "str_column": [pl.List(pl.Utf8)],
        "int_column": [
            pl.List(pl.Int64),
            pl.List(pl.Int32),
            pl.List(pl.Int16),
            pl.List(pl.Int8),
            pl.List(pl.UInt64),
            pl.List(pl.UInt32),
            pl.List(pl.UInt16),
            pl.List(pl.UInt8),
        ],
        "float_column": [pl.List(pl.Float64), pl.List(pl.Float32)],
        "bool_column": [pl.List(pl.Boolean)],
        "date_column": [pl.List(pl.Date)],
        "datetime_column": [pl.List(pl.Datetime)],
        "duration_column": [pl.List(pl.Duration)],
        "categorical_column": [pl.List(pl.Categorical), pl.List(pl.Utf8)],
        "null_column": [pl.List(pl.Null)],
    }


def test_model_joins():
    """It should produce models compatible with join statements."""

    class Left(pt.Model):
        left: int = pt.Field(gt=20)
        opt_left: Optional[int] = None

    class Right(pt.Model):
        right: int = pt.Field(gt=20)
        opt_right: Optional[int] = None

    def test_model_validator(model: Type[pt.Model]) -> None:
        """Test if all field validators have been included correctly."""
        with pytest.raises(ValidationError) as e:
            model(left=1, opt_left=1, right=1, opt_right=1)
        pattern = re.compile(r"Input should be greater than 20")
        assert len(pattern.findall(str(e.value))) == 2

    # An inner join should keep nullability information
    InnerJoinModel = Left.join(Right, how="inner")
    assert set(InnerJoinModel.columns) == {"left", "opt_left", "right", "opt_right"}
    assert InnerJoinModel.nullable_columns == {"opt_left", "opt_right"}
    assert InnerJoinModel.__name__ == "LeftInnerJoinRight"
    test_model_validator(InnerJoinModel)

    # Left joins should make all fields on left model nullable
    LeftJoinModel = Left.join(Right, how="left")
    assert set(LeftJoinModel.columns) == {"left", "opt_left", "right", "opt_right"}
    assert LeftJoinModel.nullable_columns == {"opt_left", "right", "opt_right"}
    assert LeftJoinModel.__name__ == "LeftLeftJoinRight"
    test_model_validator(LeftJoinModel)

    # Outer joins should make all columns nullable
    OuterJoinModel = Left.join(Right, how="outer")
    assert set(OuterJoinModel.columns) == {"left", "opt_left", "right", "opt_right"}
    assert OuterJoinModel.nullable_columns == {"left", "opt_left", "right", "opt_right"}
    assert OuterJoinModel.__name__ == "LeftOuterJoinRight"
    test_model_validator(OuterJoinModel)

    # Semi- and anti-joins do not change the schema at all
    assert Left.join(Right, how="semi") is Left
    assert Left.join(Right, how="anti") is Left


def test_model_selects():
    """It should produce models compatible with select statements."""

    class MyModel(pt.Model):
        a: Optional[int]
        b: int = pt.Field(gt=10)

    MySubModel = MyModel.select("b")
    assert MySubModel.columns == ["b"]
    MySubModel(b=11)
    with pytest.raises(ValidationError, match="Input should be greater than 10"):
        MySubModel(b=1)

    MyTotalModel = MyModel.select(["a", "b"])
    assert sorted(MyTotalModel.columns) == ["a", "b"]
    MyTotalModel(a=1, b=11)
    with pytest.raises(ValidationError, match="Input should be greater than 10"):
        MyTotalModel(a=1, b=1)
    assert MyTotalModel.nullable_columns == {"a"}

    with pytest.raises(
        ValueError, match="The following selected fields do not exist: {'c'}"
    ):
        MyModel.select("c")


def test_model_prefix_and_suffix():
    """It should produce models where all fields have been prefixed/suffixed."""

    class MyModel(pt.Model):
        a: Optional[int]
        b: str

    NewModel = MyModel.prefix("pre_").suffix("_post")
    assert sorted(NewModel.columns) == ["pre_a_post", "pre_b_post"]
    assert NewModel.nullable_columns == {"pre_a_post"}


def test_model_field_renaming():
    """It should be able to change its field names."""

    class MyModel(pt.Model):
        a: Optional[int]
        b: str

    NewModel = MyModel.rename({"b": "B"})
    assert sorted(NewModel.columns) == ["B", "a"]

    with pytest.raises(
        ValueError,
        match="The following fields do not exist for renaming: {'c'}",
    ):
        MyModel.rename({"c": "C"})


def test_model_field_dropping():
    """It should be able to drop a subset of its fields"""

    class MyModel(pt.Model):
        a: int
        b: int
        c: int

    assert sorted(MyModel.drop("c").columns) == ["a", "b"]
    assert MyModel.drop(["b", "c"]).columns == ["a"]


def test_with_fields():
    """It should allow whe user to add additional fields."""

    class MyModel(pt.Model):
        a: int

    ExpandedModel = MyModel.with_fields(
        b=(int, ...),
        c=(int, None),
        d=(int, pt.Field(gt=10)),
        e=(Optional[int], None),
    )
    assert sorted(ExpandedModel.columns) == list("abcde")
    assert ExpandedModel.nullable_columns == set("ce")


def test_enum_annotated_field():
    """It should use values of enums to infer types."""

    class ABCEnum(enum.Enum):
        ONE = "a"
        TWO = "b"
        THREE = "c"

    class EnumModel(pt.Model):
        column: ABCEnum

    assert EnumModel.dtypes["column"] == pl.Categorical
    assert EnumModel.example_value(field="column") == "a"
    assert EnumModel.example() == EnumModel(column="a")

    EnumModel.DataFrame({"column": ["a"]}).cast()

    class MultiTypedEnum(enum.Enum):
        ONE = 1
        TWO = "2"

    class InvalidEnumModel(pt.Model):
        column: MultiTypedEnum

    if pt._DUCKDB_AVAILABLE:  # pragma: no cover
        assert EnumModel.sql_types["column"].startswith("enum__")
        with pytest.raises(TypeError, match=r".*Encountered types: \['int', 'str'\]\."):
            InvalidEnumModel.sql_types


# TODO new tests for ColumnInfo
# def test_pt_fields():
#     class Model(pt.Model):
#         a: int
#         b: int = pt.Field(constraints=[(pl.col("b") < 10)])
#         c: int = pt.Field(derived_from=pl.col("a") + pl.col("b"))
#         d: int = pt.Field(dtype=pl.UInt8)
#         e: int = pt.Field(unique=True)

#     schema = Model.model_json_schema()  # no serialization issues
#     props = (
#         Model._schema_properties()
#     )  # extra fields are stored in modified schema_properties
#     assert "constraints" in props["b"]
#     assert "derived_from" in props["c"]
#     assert "dtype" in props["d"]
#     assert "unique" in props["e"]

#     def check_repr(field, set_value: str) -> None:
#         assert f"{set_value}=" in repr(field)
#         assert all(x not in repr(field) for x in get_args(ColumnInfo.model_fields) if x != set_value)

#     fields = (
#         Model.model_fields
#     )  # attributes are properly set and catalogued on the `FieldInfo` objects
#     assert "constraints" in fields["b"]._attributes_set
#     assert fields["b"].constraints is not None
#     check_repr(fields["b"], "constraints")
#     assert "derived_from" in fields["c"]._attributes_set
#     assert fields["c"].derived_from is not None
#     check_repr(fields["c"], "derived_from")
#     assert "dtype" in fields["d"]._attributes_set
#     assert fields["d"].dtype is not None
#     check_repr(fields["d"], "dtype")
#     assert "unique" in fields["e"]._attributes_set
#     assert fields["e"].unique is not None
#     check_repr(fields["e"], "unique")


# def test_custom_field_info():
#     class FieldExt(BaseModel):
#         foo: str | None = _Unset

#     Field = field(exts=[FieldExt])

#     class Model(pt.Model):
#         bar: int = Field(foo="hello")

#     test_field = Model.model_fields["bar"]
#     assert (
#         test_field.foo == "hello"
#     )  # TODO passes but typing is unhappy here, can we make custom FieldInfo configurable? If users subclass `Model` then it is easy to reset the typing to point at their own `FieldInfo` implementation
#     assert "foo=" in repr(test_field)
#     assert "foo" in Model._schema_properties()["bar"]
#     with pytest.raises(AttributeError):
#         print(test_field.derived_from)  # patito FieldInfo successfully overriden


def test_nullable_columns():
    class Test1(pt.Model):
        foo: str | None = pt.Field(dtype=pl.Utf8)

    assert Test1.nullable_columns == {"foo"}
    assert set(Test1.valid_dtypes["foo"]) == {pl.Utf8}

    class Test2(pt.Model):
        foo: int | None = pt.Field(dtype=pl.UInt32)

    assert Test2.nullable_columns == {"foo"}
    assert set(Test2.valid_dtypes["foo"]) == {pl.UInt32}


def test_conflicting_type_dtype():
    class Test1(pt.Model):
        foo: int = pt.Field(dtype=pl.Utf8)

    with pytest.raises(ValueError) as e:
        Test1.valid_dtypes
    assert (
        f"Invalid dtype Utf8 for column 'foo'. Allowable polars dtypes for int are: {', '.join(str(x) for x in PL_INTEGER_DTYPES)}."
        == str(e.value)
    )

    class Test2(pt.Model):
        foo: str = pt.Field(dtype=pl.Float32)

    with pytest.raises(ValueError) as e:
        Test2.valid_dtypes
    assert (
        "Invalid dtype Float32 for column 'foo'. Allowable polars dtypes for str are: Utf8."
        == str(e.value)
    )

    class Test3(pt.Model):
        foo: str | None = pt.Field(dtype=pl.UInt32)

    with pytest.raises(ValueError) as e:
        Test3.valid_dtypes
    assert (
        "Invalid dtype UInt32 for column 'foo'. Allowable polars dtypes for Union[str, NoneType] are: Utf8."
        == str(e.value)
    )


def test_polars_python_type_harmonization():
    class Test(pt.Model):
        date: datetime = pt.Field(dtype=pl.Datetime(time_unit="us"))
        # TODO add more other lesser-used type combinations here

    assert type(Test.valid_dtypes["date"][0]) == pl.Datetime
    assert Test.valid_dtypes["date"][0].time_unit == "us"
    assert Test.valid_dtypes["date"][0].time_zone is None
