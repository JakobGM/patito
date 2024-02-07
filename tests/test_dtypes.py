from datetime import date, datetime, time, timedelta
from typing import Dict, List, Literal, Sequence, Tuple

import polars as pl
import pytest
from patito._pydantic.dtypes import (
    BOOLEAN_DTYPES,
    DATE_DTYPES,
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    STRING_DTYPES,
    TIME_DTYPES,
    DataTypeGroup,
    default_polars_dtype_for_annotation,
    valid_polars_dtypes_for_annotation,
    validate_annotation,
    validate_polars_dtype,
)


def test_valids_basic_annotations():
    # base types
    assert valid_polars_dtypes_for_annotation(str) == STRING_DTYPES
    assert valid_polars_dtypes_for_annotation(int) == DataTypeGroup(
        INTEGER_DTYPES | FLOAT_DTYPES
    )
    assert valid_polars_dtypes_for_annotation(float) == FLOAT_DTYPES
    assert valid_polars_dtypes_for_annotation(bool) == BOOLEAN_DTYPES

    # temporals
    assert valid_polars_dtypes_for_annotation(datetime) == DATETIME_DTYPES
    assert valid_polars_dtypes_for_annotation(date) == DATE_DTYPES
    assert valid_polars_dtypes_for_annotation(time) == TIME_DTYPES
    assert valid_polars_dtypes_for_annotation(timedelta) == DURATION_DTYPES

    # other
    with pytest.raises(TypeError, match="must be strings"):
        valid_polars_dtypes_for_annotation(Literal[1, 2, 3])  # pyright: ignore
    with pytest.raises(TypeError, match="Mixed type enums not supported"):
        valid_polars_dtypes_for_annotation(Literal[1, 2, "3"])  # pyright: ignore

    assert valid_polars_dtypes_for_annotation(Literal["a", "b", "c"]) == {  # pyright: ignore
        pl.Categorical,
        pl.Utf8,
    }

    # combos
    assert valid_polars_dtypes_for_annotation(str | None) == STRING_DTYPES
    assert valid_polars_dtypes_for_annotation(int | float) == FLOAT_DTYPES
    assert (
        valid_polars_dtypes_for_annotation(str | int) == frozenset()
    )  # incompatible, TODO raise patito error with strict validation on

    # invalids
    assert valid_polars_dtypes_for_annotation(object) == frozenset()


def test_valids_nested_annotations():
    assert len(valid_polars_dtypes_for_annotation(List)) == 0  # needs inner annotation
    assert (
        valid_polars_dtypes_for_annotation(Tuple)
        == valid_polars_dtypes_for_annotation(List)
        == valid_polars_dtypes_for_annotation(Sequence)
    )  # for now, these are the same

    assert valid_polars_dtypes_for_annotation(List[str]) == {pl.List(pl.Utf8)}
    assert valid_polars_dtypes_for_annotation(List[str] | None) == {pl.List(pl.Utf8)}
    assert len(valid_polars_dtypes_for_annotation(List[int])) == len(
        DataTypeGroup(INTEGER_DTYPES | FLOAT_DTYPES)
    )
    assert len(valid_polars_dtypes_for_annotation(List[int | float])) == len(
        FLOAT_DTYPES
    )
    assert len(valid_polars_dtypes_for_annotation(List[int | None])) == len(
        DataTypeGroup(INTEGER_DTYPES | FLOAT_DTYPES)
    )
    assert valid_polars_dtypes_for_annotation(List[List[str]]) == {
        pl.List(pl.List(pl.Utf8))
    }  # recursion works as expected

    assert (
        valid_polars_dtypes_for_annotation(Dict) == frozenset()
    )  # not currently supported


def test_dtype_validation():
    validate_polars_dtype(int, pl.Int16)  # no issue
    validate_polars_dtype(int, pl.Float64)  # no issue
    with pytest.raises(ValueError, match="Invalid dtype"):
        validate_polars_dtype(int, pl.Utf8)

    with pytest.raises(ValueError, match="Invalid dtype"):
        validate_polars_dtype(List[str], pl.List(pl.Float64))


def test_defaults_basic_annotations():
    # base types
    assert default_polars_dtype_for_annotation(str) == pl.Utf8
    assert default_polars_dtype_for_annotation(int) == pl.Int64
    assert default_polars_dtype_for_annotation(float) == pl.Float64
    assert default_polars_dtype_for_annotation(bool) == pl.Boolean

    # temporals
    assert default_polars_dtype_for_annotation(datetime) == pl.Datetime
    assert default_polars_dtype_for_annotation(date) == pl.Date
    assert default_polars_dtype_for_annotation(time) == pl.Time
    assert default_polars_dtype_for_annotation(timedelta) == pl.Duration

    # combos
    assert default_polars_dtype_for_annotation(str | None) == pl.Utf8
    assert default_polars_dtype_for_annotation(int | float) == None
    assert default_polars_dtype_for_annotation(str | int) == None

    # invalids
    assert default_polars_dtype_for_annotation(object) == None


def test_defaults_nested_annotations():
    assert default_polars_dtype_for_annotation(List) == None  # needs inner annotation

    assert default_polars_dtype_for_annotation(List[str]) == pl.List(pl.Utf8)
    assert default_polars_dtype_for_annotation(List[str] | None) == pl.List(pl.Utf8)
    assert default_polars_dtype_for_annotation(List[int]) == pl.List(pl.Int64)
    assert default_polars_dtype_for_annotation(List[int | None]) == pl.List(pl.Int64)
    assert default_polars_dtype_for_annotation(List[int | float]) == None
    assert default_polars_dtype_for_annotation(List[str | int]) == None
    assert default_polars_dtype_for_annotation(List[List[str]]) == pl.List(
        pl.List(pl.Utf8)
    )  # recursion works as expected
    assert default_polars_dtype_for_annotation(List[List[str | None]]) == pl.List(
        pl.List(pl.Utf8)
    )

    with pytest.raises(
        ValueError, match="pydantic object types not currently supported"
    ):
        default_polars_dtype_for_annotation(Dict)


def test_annotation_validation():
    validate_annotation(int)  # no issue
    validate_annotation(int | None)
    with pytest.raises(ValueError, match="Valid dtypes are:"):
        validate_annotation(int | float)
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):
        validate_annotation(str | int)

    validate_annotation(List[int | None])
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):
        validate_annotation(List[str | int])
    with pytest.raises(ValueError, match="Valid dtypes are:"):
        validate_annotation(List[int | float])
