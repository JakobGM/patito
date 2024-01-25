"""Tests for patito.decorators"""
import pytest
import patito as pt
import polars as pl


def test_validate_hints_arg_validation_pass():
    class MyModel(pt.Model):
        a: int

    @pt.validate_hints
    def func(arg: MyModel) -> None:
        pass

    polars_dataframe = pl.DataFrame({"a": [1]})
    func(polars_dataframe)


def test_validate_hints_arg_validation_fail():
    class MyModel(pt.Model):
        a: int

    @pt.validate_hints
    def func(arg: MyModel) -> None:
        pass

    polars_dataframe = pl.DataFrame({"a": ["b"]})
    with pytest.raises(pt.ValidationError):
        func(polars_dataframe)


def test_validate_hints_return_validation_pass():
    class MyModel(pt.Model):
        a: int

    @pt.validate_hints
    def func() -> MyModel:
        return pl.DataFrame({"a": [1]})

    func()


def test_validate_hints_return_validation_fail():
    class MyModel(pt.Model):
        a: int

    @pt.validate_hints
    def func() -> MyModel:
        return pl.DataFrame({"a": ["b"]})

    with pytest.raises(pt.ValidationError):
        func()