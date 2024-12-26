"""Tests for patito.Model."""

from __future__ import annotations

# pyright: reportPrivateImportUsage=false
import enum
import re
from datetime import date, datetime, time
from typing import Optional

import polars as pl
import pytest
from polars.datatypes.group import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    DataTypeGroup,
)
from polars.testing import assert_frame_equal
from pydantic import AliasChoices, AwareDatetime, ValidationError

import patito as pt
from patito._pydantic.column_info import ColumnInfo
from patito._pydantic.dtypes.utils import (
    DATE_DTYPES,
    TIME_DTYPES,
)
from tests.examples import CompleteModel, ManyTypes, SmallModel, VerySmallModel


def test_model_example() -> None:
    """Test for Model.example()."""
    # When inheriting from Model you get a .dummy() method for generating rows with
    # default values according to the type annotation.
    SmallModel.example().model_dump()

    assert ManyTypes.example().model_dump() == {
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
        "pt_model_value": SmallModel.example().model_dump(),
        "pt_list_model_value": [VerySmallModel.example().model_dump()],
    }
    assert ManyTypes.example(
        bool_value=True,
        default_value="override",
        optional_value=1,
        pt_list_model_value=[],
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
        "pt_model_value": SmallModel.example().model_dump(),
        "pt_list_model_value": [],
    }

    ManyTypes.validate(ManyTypes.examples({"int_value": range(200)}))

    # Empty list should be valid
    ManyTypes.validate(
        ManyTypes.examples({"pt_list_model_value": [[], [VerySmallModel.example()]]})
    )

    # For now, valid regex data is not implemented
    class RegexModel(pt.Model):
        regex_column: str = pt.Field(pattern=r"[0-9a-f]")

    with pytest.raises(
        NotImplementedError,
        match="Example data generation has not been implemented for regex.*",
    ):
        RegexModel.example()


def test_model_pandas_examples() -> None:
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


def test_instantiating_model_from_row() -> None:
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


def test_instantiation_from_pandas_row() -> None:
    """You should be able to instantiate models from pandas rows."""
    pytest.importorskip("pandas")

    class Model(pt.Model):
        a: int

    polars_dataframe = pl.DataFrame({"a": [1]})
    assert Model.from_row(polars_dataframe).a == 1

    pandas_dataframe = polars_dataframe.to_pandas()
    assert Model.from_row(pandas_dataframe).a == 1
    assert Model.from_row(pandas_dataframe.loc[0]).a == 1  # type: ignore


def test_model_dataframe_class_creation() -> None:
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


def test_mapping_to_polars_dtypes() -> None:
    """Model fields should be mappable to polars dtypes."""
    assert CompleteModel.dtypes == {
        "str_column": pl.String(),
        "int_column": pl.Int64(),
        "float_column": pl.Float64(),
        "bool_column": pl.Boolean(),
        "date_column": pl.Date(),
        "datetime_column": pl.Datetime(),
        "datetime_column2": pl.Datetime(),
        "aware_datetime_column": pl.Datetime(time_zone="UTC"),
        "duration_column": pl.Duration(),
        "time_column": pl.Time(),
        "categorical_column": pl.Enum(["a", "b", "c"]),
        "null_column": pl.Null(),
        "pt_model_column": pl.Struct(
            [
                pl.Field("a", pl.Int64),
                pl.Field("b", pl.String),
                pl.Field("c", pl.Datetime(time_zone="UTC")),
                pl.Field("d", pl.Datetime(time_zone="UTC")),
                pl.Field("e", pl.Int8),
            ]
        ),
        "list_int_column": pl.List(pl.Int64),
        "list_str_column": pl.List(pl.String),
        "list_opt_column": pl.List(pl.Int64),
    }

    assert CompleteModel.valid_dtypes == {
        "str_column": {pl.String},
        "int_column": DataTypeGroup(INTEGER_DTYPES),
        "float_column": FLOAT_DTYPES,
        "bool_column": {pl.Boolean},
        "date_column": DATE_DTYPES,
        "datetime_column": DATETIME_DTYPES,
        "datetime_column2": {pl.Datetime()},
        "aware_datetime_column": {pl.Datetime(time_zone="UTC")},
        "duration_column": DURATION_DTYPES,
        "time_column": TIME_DTYPES,
        "categorical_column": {pl.Enum(["a", "b", "c"]), pl.String},
        "null_column": {pl.Null},
        "pt_model_column": DataTypeGroup(
            [
                pl.Struct(
                    [
                        pl.Field("a", pl.Int64),
                        pl.Field("b", pl.String),
                        pl.Field("c", pl.Datetime(time_zone="UTC")),
                        pl.Field("d", pl.Datetime(time_zone="UTC")),
                        pl.Field("e", pl.Int8),
                    ]
                )
            ]
        ),
        "list_int_column": DataTypeGroup(
            [pl.List(x) for x in DataTypeGroup(INTEGER_DTYPES)]
        ),
        "list_str_column": DataTypeGroup([pl.List(pl.String)]),
        "list_opt_column": DataTypeGroup(
            [pl.List(x) for x in DataTypeGroup(INTEGER_DTYPES)]
        ),
    }

    CompleteModel.example(int_column=2)
    CompleteModel.validate(CompleteModel.examples({"int_column": [1, 2, 3]}))


def test_model_joins() -> None:
    """It should produce models compatible with join statements."""

    class Left(pt.Model):
        left: int = pt.Field(gt=20)
        opt_left: Optional[int] = None

    class Right(pt.Model):
        right: int = pt.Field(gt=20)
        opt_right: Optional[int] = None

    def test_model_validator(model: type[pt.Model]) -> None:
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


def test_model_selects() -> None:
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


def test_model_prefix_and_suffix() -> None:
    """It should produce models where all fields have been prefixed/suffixed."""

    class MyModel(pt.Model):
        a: Optional[int]
        b: str

    NewModel = MyModel.prefix("pre_").suffix("_post")
    assert sorted(NewModel.columns) == ["pre_a_post", "pre_b_post"]
    assert NewModel.nullable_columns == {"pre_a_post"}


def test_model_field_renaming() -> None:
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


def test_model_field_dropping() -> None:
    """Model should be able to drop a subset of its fields."""

    class MyModel(pt.Model):
        a: int
        b: int
        c: int

    assert sorted(MyModel.drop("c").columns) == ["a", "b"]
    assert MyModel.drop(["b", "c"]).columns == ["a"]


def test_with_fields() -> None:
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


def test_enum_annotated_field() -> None:
    """It should use values of enums to infer types."""

    class ABCEnum(enum.Enum):
        ONE = "a"
        TWO = "b"
        THREE = "c"

    class EnumModel(pt.Model):
        column: ABCEnum

    assert EnumModel.dtypes["column"] == pl.Enum(["a", "b", "c"])
    assert EnumModel.example_value(field="column") == "a"
    assert EnumModel.example() == EnumModel(column="a")

    EnumModel.DataFrame({"column": ["a"]}).cast()

    class MultiTypedEnum(enum.Enum):
        ONE = 1
        TWO = "2"

    with pytest.raises(TypeError, match="Mixed type enums not supported"):

        class InvalidEnumModel(pt.Model):
            column: MultiTypedEnum

        InvalidEnumModel.validate_schema()


def test_model_schema() -> None:
    """Ensure pt.Field properties are correctly applied to model."""

    class Model(pt.Model):
        a: int = pt.Field(ge=0, unique=True)

    schema = Model.model_schema

    def validate_model_schema(schema) -> None:
        assert set(schema) == {"properties", "required", "type", "title"}
        assert schema["title"] == "Model"
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "integer"
        assert schema["properties"]["a"]["minimum"] == 0

    validate_model_schema(schema)

    # nested models
    class ParentModel(pt.Model):
        a: int
        b: Model
        c: Optional[float] = None

    schema = ParentModel.model_schema
    validate_model_schema(
        schema["$defs"]["Model"]
    )  # ensure that nested model schema is recorded in definitions
    validate_model_schema(
        schema["properties"]["b"]
    )  # and all info is copied into field properties
    assert set(schema["properties"]) == {"a", "b", "c"}
    assert schema["properties"]["a"]["required"]
    assert schema["properties"]["b"]["required"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert not schema["properties"]["c"]["required"]


def test_nullable_columns() -> None:
    """Ensure columns are correctly nullable."""

    class Test1(pt.Model):
        foo: Optional[str] = pt.Field(dtype=pl.String)

    assert Test1.nullable_columns == {"foo"}
    assert set(Test1.valid_dtypes["foo"]) == {pl.String}

    class Test2(pt.Model):
        foo: Optional[int] = pt.Field(dtype=pl.UInt32)

    assert Test2.nullable_columns == {"foo"}
    assert set(Test2.valid_dtypes["foo"]) == {pl.UInt32}


def test_conflicting_type_dtype() -> None:
    """Ensure model annotation is compatible with Field dtype."""
    with pytest.raises(ValueError, match="Invalid dtype String"):

        class Test1(pt.Model):
            foo: int = pt.Field(dtype=pl.String)

        Test1.validate_schema()

    with pytest.raises(ValueError, match="Invalid dtype Float32"):

        class Test2(pt.Model):
            foo: str = pt.Field(dtype=pl.Float32)

        Test2.validate_schema()

    with pytest.raises(ValueError, match="Invalid dtype UInt32"):

        class Test3(pt.Model):
            foo: Optional[str] = pt.Field(dtype=pl.UInt32)

        Test3.validate_schema()


def test_polars_python_type_harmonization() -> None:
    """Ensure datetime types are correctly transformed to polars types."""

    class Test(pt.Model):
        date: datetime = pt.Field(dtype=pl.Datetime(time_unit="us"))
        time: time

    assert Test.valid_dtypes["date"] == {pl.Datetime(time_unit="us")}
    assert Test.valid_dtypes["time"] == TIME_DTYPES


def test_column_infos() -> None:
    """Test that pt.Field and ColumnInfo properties match."""

    class Model(pt.Model):
        a: int
        b: int = pt.Field(constraints=[(pl.col("b") < 10)])
        c: int = pt.Field(derived_from=pl.col("a") + pl.col("b"))
        d: int = pt.Field(dtype=pl.UInt8)
        e: int = pt.Field(unique=True)

    schema = Model.model_json_schema()  # no serialization issues
    props = schema[
        "properties"
    ]  # extra fields are stored in modified schema_properties
    for col in ["b", "c", "d", "e"]:
        assert "column_info" in props[col]

    assert (
        ColumnInfo.model_validate_json(props["b"]["column_info"]).constraints
        is not None
    )
    assert (
        ColumnInfo.model_validate_json(props["c"]["column_info"]).derived_from
        is not None
    )
    assert ColumnInfo.model_validate_json(props["d"]["column_info"]).dtype is not None
    assert ColumnInfo.model_validate_json(props["e"]["column_info"]).unique is not None
    infos = Model.column_infos
    assert infos["b"].constraints is not None
    assert infos["c"].derived_from is not None
    assert infos["d"].dtype is not None
    assert infos["e"].unique is not None


def test_missing_date_struct():
    """Test model examples is validateable."""

    class SubModel(pt.Model):
        a: int
        b: AwareDatetime

    class Test(pt.Model):
        a: int
        b: int
        c: Optional[SubModel]

    df = Test.examples({"a": range(5), "c": None})
    Test.validate(df.cast())


def test_validation_alias():
    """Test that validation alias works in pt.Field.

    TODO: Not sure if this actually tests anything correctly.
    """

    class AliasModel(pt.Model):
        my_val_a: int = pt.Field(validation_alias="myValA")
        my_val_b: int = pt.Field(validation_alias=AliasChoices("my_val_b", "myValB"))

    # code from validators _find_errors showing that we need model_json_schema without aliases
    for column_name, _column_properties in AliasModel._schema_properties().items():
        assert AliasModel.column_infos[column_name] is not None
    AliasModel.examples()


def test_validation_returns_df():  # noqa: D103
    for Model in [VerySmallModel, SmallModel, ManyTypes, CompleteModel]:
        df = Model.examples()
        remade_model = Model.validate(df)
        assert_frame_equal(remade_model, df)


def test_model_iter_models():  # noqa: D103
    class SingleColumnModel(pt.Model):
        a: int

    df = SingleColumnModel.DataFrame({"a": [1, 2, 3]})

    full_list = []
    for row_model in SingleColumnModel.iter_models(df):
        assert isinstance(row_model, SingleColumnModel)
        full_list.append(row_model)
    assert len(full_list) == len(df)


def test_model_iter_models_to_list():  # noqa: D103
    class SingleColumnModel(pt.Model):
        a: int

    df = SingleColumnModel.DataFrame({"a": [1, 2, 3]})
    full_list = SingleColumnModel.iter_models(df).to_list()
    assert len(full_list) == len(df)
    for model_instance in full_list:
        assert isinstance(model_instance, SingleColumnModel)


def test_json_schema_extra_is_extended_when_it_exists() -> None:
    """Ensure that the json_schema_extra property is extended with column_info when it is set from the model field."""

    class Model(pt.Model):
        a: int
        b: int = pt.Field(
            json_schema_extra={"client_column_metadata": {"group1": "x", "group2": "y"}}
        )
        c: int = pt.Field(
            json_schema_extra={"client_column_metadata": {"group1": "xxx"}}
        )

    schema = Model.model_json_schema()  # no serialization issues
    props = schema[
        "properties"
    ]  # extra fields are stored in modified schema_properties
    for col in ["b", "c"]:
        assert "column_info" in props[col]
        assert "client_column_metadata" in props[col]
    assert "client_column_metadata" not in props["a"]
    assert props["b"]["client_column_metadata"]["group1"] == "x"
    assert props["b"]["client_column_metadata"]["group2"] == "y"
    assert props["c"]["client_column_metadata"]["group1"] == "xxx"
