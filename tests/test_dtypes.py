"""Test polars and python dtypes."""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import polars as pl
import pytest
from patito._pydantic.dtypes.dtypes import (
    DtypeResolver,
    validate_annotation,
    validate_polars_dtype,
)
from patito._pydantic.dtypes.utils import (
    BOOLEAN_DTYPES,
    DATE_DTYPES,
    STRING_DTYPES,
    TIME_DTYPES,
)
from polars.datatypes.group import (
    DATETIME_DTYPES,
    DURATION_DTYPES,
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    DataTypeGroup,
)
from pydantic import AwareDatetime

from tests.examples import ManyTypes


def test_valids_basic_annotations() -> None:
    """Test type annotations match polars dtypes."""
    # base types
    assert DtypeResolver(str).valid_polars_dtypes() == STRING_DTYPES
    assert DtypeResolver(int).valid_polars_dtypes() == DataTypeGroup(
        INTEGER_DTYPES | FLOAT_DTYPES
    )
    assert DtypeResolver(float).valid_polars_dtypes() == FLOAT_DTYPES
    assert DtypeResolver(bool).valid_polars_dtypes() == BOOLEAN_DTYPES

    # temporals
    assert DtypeResolver(datetime).valid_polars_dtypes() == DATETIME_DTYPES
    assert DtypeResolver(date).valid_polars_dtypes() == DATE_DTYPES
    assert DtypeResolver(time).valid_polars_dtypes() == TIME_DTYPES
    assert DtypeResolver(timedelta).valid_polars_dtypes() == DURATION_DTYPES

    # other
    with pytest.raises(TypeError, match="must be strings"):
        DtypeResolver(Literal[1, 2, 3]).valid_polars_dtypes()  # pyright: ignore
    with pytest.raises(TypeError, match="Mixed type enums not supported"):
        DtypeResolver(Literal[1, 2, "3"]).valid_polars_dtypes()  # pyright: ignore

    assert DtypeResolver(Literal["a", "b", "c"]).valid_polars_dtypes() == {  # pyright: ignore
        pl.Enum(["a", "b", "c"]),
        pl.String,
    }

    # combos
    assert DtypeResolver(Optional[str]).valid_polars_dtypes() == STRING_DTYPES
    if sys.version_info[1] >= 10:
        assert (
            DtypeResolver(str | None | None).valid_polars_dtypes() == STRING_DTYPES
        )  # superfluous None is ok
    assert DtypeResolver(Union[int, float]).valid_polars_dtypes() == FLOAT_DTYPES
    assert (
        DtypeResolver(Union[str, int]).valid_polars_dtypes() == frozenset()
    )  # incompatible

    # invalids
    assert DtypeResolver(object).valid_polars_dtypes() == frozenset()


def test_valids_nested_annotations() -> None:
    """Test type annotations match nested polars types like List."""
    assert len(DtypeResolver(List).valid_polars_dtypes()) == 0  # needs inner annotation
    assert (
        DtypeResolver(Tuple).valid_polars_dtypes()
        == DtypeResolver(List).valid_polars_dtypes()
        == DtypeResolver(Sequence).valid_polars_dtypes()
    )  # for now, these are the same

    assert DtypeResolver(List[str]).valid_polars_dtypes() == {pl.List(pl.String)}
    assert DtypeResolver(Optional[List[str]]).valid_polars_dtypes() == {
        pl.List(pl.String)
    }
    assert len(DtypeResolver(List[int]).valid_polars_dtypes()) == len(
        DataTypeGroup(INTEGER_DTYPES | FLOAT_DTYPES)
    )
    assert len(DtypeResolver(List[Union[int, float]]).valid_polars_dtypes()) == len(
        FLOAT_DTYPES
    )
    assert len(DtypeResolver(List[Optional[int]]).valid_polars_dtypes()) == len(
        DataTypeGroup(INTEGER_DTYPES | FLOAT_DTYPES)
    )
    assert DtypeResolver(List[List[str]]).valid_polars_dtypes() == {
        pl.List(pl.List(pl.String))
    }  # recursion works as expected

    assert (
        DtypeResolver(Dict).valid_polars_dtypes() == frozenset()
    )  # not currently supported

    # support for nested models via struct
    assert (
        len(DtypeResolver(ManyTypes).valid_polars_dtypes()) == 1
    )  # only defaults are valid
    assert (
        DtypeResolver(ManyTypes).valid_polars_dtypes()
        == DtypeResolver(Optional[ManyTypes]).valid_polars_dtypes()
    )


def test_dtype_validation() -> None:
    """Ensure python types match polars types."""
    validate_polars_dtype(int, pl.Int16)  # no issue
    validate_polars_dtype(int, pl.Float64)  # no issue
    with pytest.raises(ValueError, match="Invalid dtype"):
        validate_polars_dtype(int, pl.String)

    with pytest.raises(ValueError, match="Invalid dtype"):
        validate_polars_dtype(List[str], pl.List(pl.Float64))

    # some potential corner cases
    validate_polars_dtype(AwareDatetime, dtype=pl.Datetime(time_zone="UTC"))


def test_defaults_basic_annotations() -> None:
    """Ensure python types resolve to largest polars type."""
    # base types
    assert DtypeResolver(str).default_polars_dtype() == pl.String
    assert DtypeResolver(int).default_polars_dtype() == pl.Int64
    assert DtypeResolver(float).default_polars_dtype() == pl.Float64
    assert DtypeResolver(bool).default_polars_dtype() == pl.Boolean

    # temporals
    assert DtypeResolver(datetime).default_polars_dtype() == pl.Datetime
    assert DtypeResolver(date).default_polars_dtype() == pl.Date
    assert DtypeResolver(time).default_polars_dtype() == pl.Time
    assert DtypeResolver(timedelta).default_polars_dtype() == pl.Duration

    # combos
    assert DtypeResolver(Optional[str]).default_polars_dtype() == pl.String
    assert DtypeResolver(Union[int, float]).default_polars_dtype() is None
    assert DtypeResolver(Union[str, int]).default_polars_dtype() is None

    # other
    literal = DtypeResolver(Literal["a", "b", "c"]).default_polars_dtype()
    assert literal == pl.Enum(["a", "b", "c"])
    assert set(literal.categories) == {"a", "b", "c"}

    # invalids
    assert DtypeResolver(object).default_polars_dtype() is None


def test_defaults_nested_annotations() -> None:
    """Ensure python nested types fallback to largest nested polars type."""
    assert DtypeResolver(List).default_polars_dtype() is None  # needs inner annotation

    assert DtypeResolver(List[str]).default_polars_dtype() == pl.List(pl.String)
    assert DtypeResolver(Optional[List[str]]).default_polars_dtype() == pl.List(
        pl.String
    )
    assert DtypeResolver(List[int]).default_polars_dtype() == pl.List(pl.Int64)
    assert DtypeResolver(List[Optional[int]]).default_polars_dtype() == pl.List(
        pl.Int64
    )
    assert DtypeResolver(List[Union[int, float]]).default_polars_dtype() is None
    assert DtypeResolver(List[Union[str, int]]).default_polars_dtype() is None
    assert DtypeResolver(List[List[str]]).default_polars_dtype() == pl.List(
        pl.List(pl.String)
    )  # recursion works as expected
    assert DtypeResolver(List[List[Optional[str]]]).default_polars_dtype() == pl.List(
        pl.List(pl.String)
    )

    with pytest.raises(
        NotImplementedError, match="dictionaries not currently supported"
    ):
        DtypeResolver(Dict).default_polars_dtype()

    # support for nested models via struct
    many_types = DtypeResolver(ManyTypes).default_polars_dtype()
    assert many_types == pl.Struct
    assert len(many_types.fields) == len(ManyTypes.columns)
    assert DtypeResolver(Optional[ManyTypes]).default_polars_dtype() == many_types


def test_annotation_validation() -> None:
    """Check that python types are resolveable."""
    validate_annotation(int)  # no issue
    validate_annotation(Optional[int])

    with pytest.raises(ValueError, match="Valid dtypes are:"):
        validate_annotation(Union[int, float])
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):
        validate_annotation(Union[str, int])

    validate_annotation(List[Optional[int]])
    with pytest.raises(ValueError, match="not compatible with any polars dtypes"):
        validate_annotation(List[Union[str, int]])
    with pytest.raises(ValueError, match="Valid dtypes are:"):
        validate_annotation(List[Union[int, float]])
