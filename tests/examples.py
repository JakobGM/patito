"""Testing examples."""

from datetime import date, datetime, time, timedelta
from typing import Literal, Optional

import polars as pl
from pydantic import AwareDatetime

import patito as pt


class VerySmallModel(pt.Model):
    """Very small model for testing."""

    a: int
    b: str


class SmallModel(pt.Model):
    """Small model for testing."""

    a: int
    b: str
    c: AwareDatetime = pt.Field(
        dtype=pl.Datetime(time_zone="UTC")
    )  # check that dtype resolver will use patito-specified dtype if passed
    d: Optional[AwareDatetime] = pt.Field(
        default=None, dtype=pl.Datetime(time_zone="UTC")
    )
    e: int = pt.Field(dtype=pl.Int8)


class ManyTypes(pt.Model):
    """Medium model for testing."""

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
    pt_model_value: SmallModel
    pt_list_model_value: list[VerySmallModel]


class CompleteModel(pt.Model):
    """Model containing all combinations of python types."""

    str_column: str
    int_column: int
    float_column: float
    bool_column: bool

    date_column: date
    datetime_column: datetime
    datetime_column2: datetime = pt.Field(dtype=pl.Datetime)
    aware_datetime_column: AwareDatetime = pt.Field(dtype=pl.Datetime(time_zone="UTC"))
    duration_column: timedelta
    time_column: time

    categorical_column: Literal["a", "b", "c"]
    null_column: None = None

    pt_model_column: SmallModel

    list_int_column: list[int]
    list_str_column: list[str]
    list_opt_column: list[Optional[int]]
