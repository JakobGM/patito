"""Tests for the patito.validators module."""

from __future__ import annotations

import enum
import re
import sys
from datetime import date, datetime
from typing import List, Literal, Optional, Union

import patito as pt
import polars as pl
import pytest
from patito import DataFrameValidationError
from patito._pydantic.dtypes import is_optional
from patito.validators import _dewrap_optional, validate
from pydantic.aliases import AliasGenerator
from pydantic.config import ConfigDict


def test_is_optional() -> None:
    """It should return True for optional types."""
    assert is_optional(Optional[int])
    assert is_optional(Union[int, None])
    assert not is_optional(int)


@pytest.mark.skipif(
    sys.version_info <= (3, 10),
    reason="Using | as a type union operator is only supported from python 3.10.",
)
def test_is_optional_with_pipe_operator() -> None:
    """It should return True for optional types."""
    assert is_optional(int | None)  # typing: ignore  # pragma: noqa  # pyright: ignore


def test_dewrap_optional() -> None:
    """It should return the inner type of Optional types."""
    assert _dewrap_optional(Optional[int]) is int
    assert _dewrap_optional(Union[int, None]) is int
    assert _dewrap_optional(int) is int


@pytest.mark.skipif(
    sys.version_info <= (3, 10),
    reason="Using | as a type union operator is only supported from python 3.10.",
)
def test_dewrap_optional_with_pipe_operator() -> None:
    """It should return the inner type of Optional types."""
    assert (  # typing: ignore  # pragma: noqa  # pyright: ignore
        _dewrap_optional(int | None) is int
    )


def test_missing_column_validation() -> None:
    """Validation should catch missing columns."""

    class SingleColumnModel(pt.Model):
        column_1: int
        column_2: str

    # First we raise an error because we are missing column_1
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=pl.DataFrame(), schema=SingleColumnModel)

    errors = e_info.value.errors()
    assert len(e_info.value.errors()) == 2
    assert sorted(errors, key=lambda e: e["loc"]) == [
        {
            "loc": ("column_1",),
            "msg": "Missing column",
            "type": "type_error.missingcolumns",
        },
        {
            "loc": ("column_2",),
            "msg": "Missing column",
            "type": "type_error.missingcolumns",
        },
    ]

    df_missing_column_2 = pl.DataFrame({"column_1": [1, 2, 3]})
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=df_missing_column_2, schema=SingleColumnModel)
    validate(
        dataframe=df_missing_column_2,
        schema=SingleColumnModel,
        allow_missing_columns=True,
    )  # does not raise when allow_missing_columns=True
    SingleColumnModel.validate(
        df_missing_column_2, allow_missing_columns=True
    )  # kwargs are passed via model-centric validation API


def test_superfluous_column_validation() -> None:
    """Validation should catch superfluous columns."""

    class SingleColumnModel(pt.Model):
        column_1: int

    # We raise an error because we have added column_2
    test_df = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("column_1"),
            pl.lit(2).alias("column_2"),
        ]
    )
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(
            dataframe=test_df,
            schema=SingleColumnModel,
        )

    errors = e_info.value.errors()
    assert len(e_info.value.errors()) == 1
    assert errors[0] == {
        "loc": ("column_2",),
        "msg": "Superfluous column",
        "type": "type_error.superfluouscolumns",
    }

    validate(
        test_df, SingleColumnModel, allow_superfluous_columns=True
    )  # does not raise
    SingleColumnModel.validate(
        test_df, allow_superfluous_columns=True
    )  # model-centric API also works


def test_validate_non_nullable_columns() -> None:
    """Test for validation logic related to missing values."""

    class SmallModel(pt.Model):
        column_1: int
        column_2: Optional[int] = None

    # We insert nulls into a non-optional column, causing an exception
    wrong_nulls_df = pl.DataFrame().with_columns(
        [
            pl.lit(None).cast(pl.Int64).alias("column_1"),
            pl.lit(None).cast(pl.Int64).alias("column_2"),
        ]
    )
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(
            dataframe=wrong_nulls_df,
            schema=SmallModel,
        )

    errors = e_info.value.errors()
    assert len(e_info.value.errors()) == 1
    assert errors[0] == {
        "loc": ("column_1",),
        "msg": "1 missing value",
        "type": "value_error.missingvalues",
    }


def test_validate_dtype_checks() -> None:
    """Test dtype-checking logic."""

    # An integer field may be validated against several different integer dtypes
    class IntModel(pt.Model):
        column: int

    for dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        series = pl.Series([], dtype=dtype).alias("column")
        dataframe = pl.DataFrame([series])
        validate(dataframe=dataframe, schema=IntModel)

    # But other types must be considered invalid
    for dtype in (pl.String, pl.Date):
        series = pl.Series([], dtype=dtype).alias("column")
        dataframe = pl.DataFrame([series])
        with pytest.raises(DataFrameValidationError) as e_info:
            validate(dataframe=dataframe, schema=IntModel)

        errors = e_info.value.errors()
        assert len(e_info.value.errors()) == 1
        assert errors[0] == {
            "loc": ("column",),
            "msg": f"Polars dtype {dtype} does not match model field type.",
            "type": "type_error.columndtype",
        }

    # We construct a model with most of the common field types
    class CompleteModel(pt.Model):
        int_column: int
        string_column: str
        float_column: float
        date_column: date
        datetime_column: datetime
        bool_column: bool

    # And validate it againt a valid dataframe
    valid_df = pl.DataFrame().with_columns(
        [
            pl.lit(1, dtype=pl.Int16).alias("int_column"),
            pl.lit("a", dtype=pl.String).alias("string_column"),
            pl.lit(1.0, dtype=pl.Float32).alias("float_column"),
            pl.lit(datetime.now(), dtype=pl.Datetime).alias("datetime_column"),
            pl.lit(date.today(), dtype=pl.Date).alias("date_column"),
            pl.lit(True, dtype=pl.Boolean).alias("bool_column"),
        ]
    )
    validate(dataframe=valid_df, schema=CompleteModel)

    # We try to hit each column dtype check
    for column in CompleteModel.columns:
        if column == "int_column":
            dtype = pl.String
        else:
            dtype = pl.Int64

        with pytest.raises(DataFrameValidationError) as e_info:
            validate(
                dataframe=valid_df.with_columns(pl.lit(1, dtype=dtype).alias(column)),
                schema=CompleteModel,
            )

        errors = e_info.value.errors()
        assert len(e_info.value.errors()) == 1

        assert errors[0] == {
            "loc": (column,),
            "msg": f"Polars dtype {dtype} does not match model field type.",
            "type": "type_error.columndtype",
        }

    # Anything non-compatible with polars should raise NotImplementedError
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):

        class NonCompatibleModel(pt.Model):
            my_field: object

        NonCompatibleModel.validate_schema()

    # The same goes for list-annotated fields
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):

        class NonCompatibleListModel(pt.Model):
            my_field: List[object]

        NonCompatibleListModel.validate_schema()

    # It should also work with pandas data frames
    class PandasCompatibleModel(CompleteModel):
        date_column: str  # type: ignore

    pytest.importorskip("pandas")
    validate(
        dataframe=valid_df.with_columns(pl.col("date_column").cast(str)).to_pandas(),
        schema=PandasCompatibleModel,
    )


def test_uniqueness_validation() -> None:
    """It should be able to validate uniqueness."""

    class MyModel(pt.Model):
        column: int = pt.Field(unique=True)

    non_duplicated_df = pt.DataFrame({"column": [1, 2, 3]})
    MyModel.validate(non_duplicated_df)

    empty_df = pt.DataFrame([pl.Series("column", [], dtype=pl.Int64)])
    MyModel.validate(empty_df)

    duplicated_df = pt.DataFrame({"column": [1, 1, 2]})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        MyModel.validate(duplicated_df)


def test_datetime_validation() -> None:
    """Test for date(time) validation.

    Both strings, dates, and datetimes are assigned type "string" in the OpenAPI JSON
    schema spec, so this needs to be specifically tested for since the implementation
    needs to check the "format" property on the field schema.
    """
    string_df = pl.DataFrame().with_columns(
        pl.lit("string", dtype=pl.String).alias("c")
    )
    date_df = pl.DataFrame().with_columns(
        pl.lit(date.today(), dtype=pl.Date).alias("c")
    )
    datetime_df = pl.DataFrame().with_columns(
        pl.lit(datetime.now(), dtype=pl.Datetime).alias("c")
    )

    class StringModel(pt.Model):
        c: str

    validate(dataframe=string_df, schema=StringModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=date_df, schema=StringModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=datetime_df, schema=StringModel)

    class DateModel(pt.Model):
        c: date

    validate(dataframe=date_df, schema=DateModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=string_df, schema=DateModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=datetime_df, schema=DateModel)

    class DateTimeModel(pt.Model):
        c: datetime

    validate(dataframe=datetime_df, schema=DateTimeModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=string_df, schema=DateTimeModel)
    with pytest.raises(DataFrameValidationError):
        validate(dataframe=date_df, schema=DateTimeModel)


def test_enum_validation() -> None:
    """Test validation of enum.Enum-typed fields."""

    class ABCEnum(enum.Enum):
        ONE = "a"
        TWO = "b"
        THREE = "c"

    class EnumModel(pt.Model):
        column: ABCEnum

    valid_df = pl.DataFrame({"column": ["a", "b", "b", "c"]})
    validate(dataframe=valid_df, schema=EnumModel)

    invalid_df = pl.DataFrame({"column": ["d"]})
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=invalid_df, schema=EnumModel)

    errors = e_info.value.errors()
    assert len(errors) == 1
    assert errors[0] == {
        "loc": ("column",),
        "msg": "Rows with invalid values: {'d'}.",
        "type": "value_error.rowvalue",
    }


def test_literal_enum_validation() -> None:
    """Test validation of typing.Literal-typed fields."""

    class EnumModel(pt.Model):
        column: Literal["a", "b", "c"]

    valid_df = pl.DataFrame({"column": ["a", "b", "b", "c"]})
    validate(dataframe=valid_df, schema=EnumModel)

    invalid_df = pl.DataFrame({"column": ["d"]})
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=invalid_df, schema=EnumModel)

    error_expected = {
        "loc": ("column",),
        "msg": "Rows with invalid values: {'d'}.",
        "type": "value_error.rowvalue",
    }
    errors = e_info.value.errors()
    assert len(errors) == 1
    assert errors[0] == error_expected

    class ListEnumModel(pt.Model):
        column: List[Literal["a", "b", "c"]]

    valid_df = pl.DataFrame({"column": [["a", "b"], ["b", "c"], ["a", "c"]]})
    validate(dataframe=valid_df, schema=ListEnumModel)

    invalid_df = pl.DataFrame({"column": [["a", "b"], ["b", "c"], ["a", "d"]]})
    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=invalid_df, schema=ListEnumModel)
    errors = e_info.value.errors()
    assert len(errors) == 1
    assert errors[0] == error_expected


class _PositiveStruct(pt.Model):
    x: int = pt.Field(gt=0)


class _PositiveStructModel(pt.Model):
    positive_struct: _PositiveStruct


def test_simple_struct_validation() -> None:
    """Test validation of model with struct column."""
    valid_df = pl.DataFrame({"positive_struct": [{"x": 1}, {"x": 2}, {"x": 3}]})
    _PositiveStructModel.validate(valid_df)

    bad_df = pl.DataFrame({"positive_struct": [{"x": -1}, {"x": 2}, {"x": 3}]})
    with pytest.raises(DataFrameValidationError):
        _PositiveStructModel.validate(bad_df)


def test_nested_struct_validation() -> None:
    """Test validation of model with nested struct column."""

    class NestedPositiveStructModel(pt.Model):
        positive_struct_model: _PositiveStructModel

    valid_df = pl.DataFrame(
        {
            "positive_struct_model": [
                {"positive_struct": {"x": 1}},
                {"positive_struct": {"x": 2}},
                {"positive_struct": {"x": 3}},
            ]
        }
    )
    NestedPositiveStructModel.validate(valid_df)

    bad_df = pl.DataFrame(
        {
            "positive_struct_model": [
                {"positive_struct": {"x": -1}},
                {"positive_struct": {"x": 2}},
                {"positive_struct": {"x": 3}},
            ]
        }
    )
    with pytest.raises(DataFrameValidationError):
        NestedPositiveStructModel.validate(bad_df)


def test_list_struct_validation() -> None:
    """Test validation of model with list of structs column."""

    class ListPositiveStructModel(pt.Model):
        list_positive_struct: list[_PositiveStruct]

    valid_df = pl.DataFrame(
        {"list_positive_struct": [[{"x": 1}, {"x": 2}], [{"x": 3}, {"x": 4}, {"x": 5}]]}
    )
    ListPositiveStructModel.validate(valid_df)

    bad_df = pl.DataFrame(
        {
            "list_positive_struct": [
                [{"x": 1}, {"x": 2}],
                [{"x": 3}, {"x": -4}, {"x": 5}],
            ]
        }
    )
    with pytest.raises(DataFrameValidationError):
        ListPositiveStructModel.validate(bad_df)


def test_struct_validation_with_polars_constraint() -> None:
    """Test validation of models with constrained struct column."""

    class Interval(pt.Model):
        x_min: int
        x_max: int = pt.Field(constraints=pt.col("x_min") <= pt.col("x_max"))

    class IntervalModel(pt.Model):
        interval: Interval

    valid_df = pl.DataFrame(
        {
            "interval": [
                {"x_min": 0, "x_max": 1},
                {"x_min": 0, "x_max": 0},
                {"x_min": -1, "x_max": 1},
            ]
        }
    )
    IntervalModel.validate(valid_df)

    bad_df = pl.DataFrame(
        {
            "interval": [
                {"x_min": 0, "x_max": 1},
                {"x_min": 1, "x_max": 0},
                {"x_min": -1, "x_max": 1},
            ]
        }
    )
    with pytest.raises(DataFrameValidationError):
        IntervalModel.validate(bad_df)


def test_uniqueness_constraint_validation() -> None:
    """Uniqueness constraints should be validated."""

    class UniqueModel(pt.Model):
        product_id: int = pt.Field(unique=True)

    validate(dataframe=pl.DataFrame({"product_id": [1, 2]}), schema=UniqueModel)

    with pytest.raises(DataFrameValidationError) as e_info:
        validate(dataframe=pl.DataFrame({"product_id": [1, 1]}), schema=UniqueModel)

    errors = e_info.value.errors()
    assert len(errors) == 1
    assert errors[0] == {
        "loc": ("product_id",),
        "msg": "2 rows with duplicated values.",
        "type": "value_error.rowvalue",
    }


def test_validation_of_bounds_checks() -> None:
    """Check if value bounds are correctly validated."""

    class BoundModel(pt.Model):
        le_column: float = pt.Field(le=42.5)
        lt_column: float = pt.Field(lt=42.5)
        ge_column: float = pt.Field(ge=42.5)
        gt_column: float = pt.Field(gt=42.5)
        combined_column: float = pt.Field(gt=42.5, le=43)
        multiple_column: float = pt.Field(multiple_of=0.5)
        # const fields should now use Literal instead, but pyright
        # complains about Literal of float values
        const_column: Literal['3.1415'] = pt.Field(default='3.1415')  # type: ignore
        regex_column: str = pt.Field(pattern=r"value [A-Z]")
        min_length_column: str = pt.Field(min_length=2)
        max_length_column: str = pt.Field(max_length=2)

    # The .example() method should produce the correct dtypes, except for
    # the regex-validated string field which is not supported
    BoundModel.validate(
        BoundModel.examples({"regex_column": ["value A", "value B", "value C"]})
    )

    valid = [42.5, 42.4, 42.5, 42.6, 42.6, 19.5, '3.1415', "value X", "ab", "ab"]
    valid_df = pl.DataFrame(data=[valid], schema=BoundModel.columns, orient='row')
    BoundModel.validate(valid_df)

    invalid = [42.6, 42.5, 42.4, 42.5, 43.1, 19.75, '3.2', "value x", "a", "abc"]
    for column_index, column_name in enumerate(BoundModel.columns):
        data = (
            valid[:column_index]
            + invalid[column_index : column_index + 1]
            + valid[column_index + 1 :]
        )
        invalid_df = pl.DataFrame(data=[data], schema=BoundModel.columns, orient='row')
        with pytest.raises(DataFrameValidationError) as e_info:
            BoundModel.validate(invalid_df)
        errors = e_info.value.errors()

        if column_name == 'const_column':
            assert len(errors) == 2
        else:

            assert len(errors) == 1
            assert errors[0] == {
                "loc": (column_name,),
                "msg": "1 row with out of bound values.",
                "type": "value_error.rowvalue",
            }


def test_validation_of_dtype_specifiers() -> None:
    """Fields with specific dtype annotations should be validated."""

    class DTypeModel(pt.Model):
        int_column: int
        int_explicit_dtype_column: int = pt.Field(dtype=pl.Int64)
        smallint_column: int = pt.Field(dtype=pl.Int8)
        unsigned_int_column: int = pt.Field(dtype=pl.UInt64)
        unsigned_smallint_column: int = pt.Field(dtype=pl.UInt8)

    assert DTypeModel.dtypes == {
        "int_column": pl.Int64,
        "int_explicit_dtype_column": pl.Int64,
        "smallint_column": pl.Int8,
        "unsigned_int_column": pl.UInt64,
        "unsigned_smallint_column": pl.UInt8,
    }

    # The .example() method should produce the correct dtypes
    DTypeModel.validate(DTypeModel.examples({"smallint_column": [1, 2, 3]}))

    valid = [
        pl.Series([-2]).cast(pl.Int64),
        pl.Series([2**32]).cast(pl.Int64),
        pl.Series([2]).cast(pl.Int8),
        pl.Series([2]).cast(pl.UInt64),
        pl.Series([2]).cast(pl.UInt8),
    ]
    valid_df = pl.DataFrame(data=valid, schema=DTypeModel.columns)
    DTypeModel.validate(valid_df)

    invalid = [
        pl.Series(["a"]).cast(pl.String),
        pl.Series([2.5]).cast(pl.Float64),
        pl.Series([2**32]).cast(pl.Int64),
        pl.Series([-2]).cast(pl.Int64),
        pl.Series([-2]).cast(pl.Int64),
    ]
    for column_index, (column_name, dtype) in enumerate(
        zip(
            DTypeModel.columns,
            [pl.String, pl.Float64, pl.Int64, pl.Int64, pl.Int64],
        )
    ):
        data = (
            valid[:column_index]
            + invalid[column_index : column_index + 1]
            + valid[column_index + 1 :]
        )
        invalid_df = pl.DataFrame(data=data, schema=DTypeModel.columns)
        with pytest.raises(DataFrameValidationError) as e_info:
            DTypeModel.validate(invalid_df)
        errors = e_info.value.errors()
        assert len(errors) == 1
        assert errors[0] == {
            "loc": (column_name,),
            "msg": f"Polars dtype {dtype} does not match model field type.",
            "type": "type_error.columndtype",
        }


def test_custom_constraint_validation() -> None:
    """Users should be able to specify custom constraints."""

    class CustomConstraintModel(pt.Model):
        even_int: int = pt.Field(
            constraints=[(pl.col("even_int") % 2 == 0).alias("even_constraint")]
        )
        odd_int: int = pt.Field(constraints=pl.col("odd_int") % 2 == 1)

    df = CustomConstraintModel.DataFrame({"even_int": [2, 3], "odd_int": [1, 2]})
    with pytest.raises(DataFrameValidationError) as e_info:
        df.validate()
    errors = e_info.value.errors()
    assert len(errors) == 2
    assert errors[0] == {
        "loc": ("even_int",),
        "msg": "1 row does not match custom constraints.",
        "type": "value_error.rowvalue",
    }
    assert errors[1] == {
        "loc": ("odd_int",),
        "msg": "1 row does not match custom constraints.",
        "type": "value_error.rowvalue",
    }
    df.limit(1).validate()

    # We can also validate aggregation queries
    class PizzaSlice(pt.Model):
        fraction: float = pt.Field(constraints=pl.col("fraction").sum() == 1)

    whole_pizza = pt.DataFrame({"fraction": [0.25, 0.75]})
    PizzaSlice.validate(whole_pizza)

    part_pizza = pt.DataFrame({"fraction": [0.25, 0.25]})
    with pytest.raises(DataFrameValidationError):
        PizzaSlice.validate(part_pizza)

    # We can validate multiple AND constraints with a list of constraints
    class DivisibleByTwoAndThree(pt.Model):
        number: int = pt.Field(constraints=[pt.col("_") % 2 == 0, pt.col("_") % 3 == 0])

    one_constraint_failing_df = pt.DataFrame({"number": [3]})
    with pytest.raises(DataFrameValidationError):
        DivisibleByTwoAndThree.validate(one_constraint_failing_df)

    other_constraint_failing_df = pt.DataFrame({"number": [4]})
    with pytest.raises(DataFrameValidationError):
        DivisibleByTwoAndThree.validate(other_constraint_failing_df)

    all_constraints_failing_df = pt.DataFrame({"number": [5]})
    with pytest.raises(DataFrameValidationError):
        DivisibleByTwoAndThree.validate(all_constraints_failing_df)

    all_constraints_passing_df = pt.DataFrame({"number": [6]})
    DivisibleByTwoAndThree.validate(all_constraints_passing_df)


def test_anonymous_column_constraints() -> None:
    """You should be able to refer to the field column with an anonymous column."""

    class Pair(pt.Model):
        # pl.col("_") refers to the given field column
        odd_number: int = pt.Field(constraints=pl.col("_") % 2 == 1)
        # pt.field is simply an alias for pl.col("_")
        even_number: int = pt.Field(constraints=pt.field % 2 == 0)

    pairs = pt.DataFrame({"odd_number": [1, 3, 5], "even_number": [2, 4, 6]})
    Pair.validate(pairs)
    with pytest.raises(DataFrameValidationError):
        Pair.validate(
            pairs.select(
                [
                    pl.col("odd_number").alias("even_number"),
                    pl.col("even_number").alias("odd_number"),
                ]
            )
        )


def test_optional_enum() -> None:
    """It should handle optional enums correctly."""

    class OptionalEnumModel(pt.Model):
        # Old type annotation syntax
        optional_enum: Optional[Literal["A", "B"]]

    df = pl.DataFrame({"optional_enum": ["A", "B", None]})
    OptionalEnumModel.validate(df)


@pytest.mark.skipif(
    sys.version_info <= (3, 10),
    reason="Using | as a type union operator is only supported from python 3.10.",
)
def test_optional_pipe_operator() -> None:
    """Ensure that pipe operator works as expected."""

    class OptionalEnumModel(pt.Model):
        # Old type annotation syntax
        optional_enum_1: Optional[Literal["A", "B"]]
        # New type annotation syntax
        optional_enum_2: Literal["A", "B"] | None  # type: ignore

    df = pl.DataFrame(
        {
            "optional_enum_1": ["A", "B", None],
            "optional_enum_2": ["A", "B", None],
        }
    )
    OptionalEnumModel.validate(df)


@pytest.mark.xfail(
    condition=sys.version_info < (3, 8),
    reason="Polars bug: list series can not be constructed with None values.",
    raises=TypeError,
    strict=True,
)
def test_validation_of_list_dtypes() -> None:
    """It should be able to validate dtypes organized in lists."""

    class ListModel(pt.Model):
        int_list: List[int]
        int_or_null_list: List[Optional[int]]
        nullable_int_list: Optional[List[int]]
        nullable_int_or_null_list: Optional[List[Optional[int]]]

    valid_df = pl.DataFrame(
        {
            "int_list": [[1, 2], [3, 4]],
            "int_or_null_list": [[1, 2], [3, None]],
            "nullable_int_list": [[1, 2], None],
            "nullable_int_or_null_list": [[1, None], None],
        }
    )
    ListModel.validate(valid_df)

    for old, new in [
        # List items are not nullable
        ("int_or_null_list", "int_list"),
        ("int_or_null_list", "nullable_int_list"),
        # List is not nullable
        ("nullable_int_list", "int_list"),
        ("nullable_int_list", "int_or_null_list"),
        # Combination of both
        ("nullable_int_or_null_list", "int_list"),
        ("nullable_int_or_null_list", "int_or_null_list"),
        ("nullable_int_or_null_list", "nullable_int_list"),
    ]:
        with pytest.raises(DataFrameValidationError):
            ListModel.validate(valid_df.with_columns(pl.col(old).alias(new)))


def test_nested_field_attrs() -> None:
    """Ensure that constraints are respected even when embedded inside 'anyOf'."""

    class Test(pt.Model):
        foo: Optional[int] = pt.Field(
            dtype=pl.Int64, ge=0, le=100, constraints=pt.field.sum() == 100
        )

    test_df = Test.DataFrame(
        {"foo": [110, -10]}
    )  # meets constraint, but violates bounds (embedded in 'anyOf' in properties)
    with pytest.raises(DataFrameValidationError) as e:
        Test.validate(test_df)
    pattern = re.compile(r"2 rows with out of bound values")
    assert len(pattern.findall(str(e.value))) == 1

    null_test_df = Test.DataFrame({"foo": [100, None, None]})
    Test.validate(null_test_df)  # should not raise


def test_validation_column_subset() -> None:
    """Ensure that columns are only validated if they are in the subset."""

    class Test(pt.Model):
        a: int
        b: int = pt.Field(dtype=pl.Int64, ge=0, le=100)

    Test.validate(pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}))  # should pass
    with pytest.raises(DataFrameValidationError):
        Test.validate(pl.DataFrame({"a": [1, 2, 3], "b": [101, 102, 103]}))

    # should pass without validating b
    Test.validate(pl.DataFrame({"a": [1, 2, 3], "b": [101, 102, 103]}), columns=["a"])

    with pytest.raises(DataFrameValidationError):
        Test.validate(
            pl.DataFrame({"a": [1, 2, 3], "b": [101, 102, 103]}), columns=["b"]
        )
    # test asking for superfluous column
    with pytest.raises(DataFrameValidationError):
        Test.validate(pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), columns=["c"])


def test_alias_generator() -> None:
    """Allow column name transformations through AliasGenerator."""
    df = pl.DataFrame({"my_val_a": [0]})

    class NoAliasGeneratorModel(pt.Model):
        My_Val_A: int

    with pytest.raises(DataFrameValidationError):
        NoAliasGeneratorModel.validate(df)

    class AliasGeneratorModel(pt.Model):
        model_config = ConfigDict(
            alias_generator=AliasGenerator(validation_alias=str.title),
        )
        My_Val_A: int

    AliasGeneratorModel.validate(df)

    df = pl.DataFrame({"my_incorrect_val_a": [0]})
    with pytest.raises(DataFrameValidationError):
        AliasGeneratorModel.validate(df)


def test_alias_generator_func() -> None:
    """Allow column name transformations through a string function."""
    df = pl.DataFrame({"my_val_a": [0]})

    class NoAliasGeneratorModel(pt.Model):
        My_Val_A: int

    with pytest.raises(DataFrameValidationError):
        NoAliasGeneratorModel.validate(df)

    class AliasGeneratorModel(pt.Model):
        model_config = ConfigDict(
            alias_generator=str.title,
        )
        My_Val_A: int

    AliasGeneratorModel.validate(df)

    df = pl.DataFrame({"my_incorrect_val_a": [0]})
    with pytest.raises(DataFrameValidationError):
        AliasGeneratorModel.validate(df)
