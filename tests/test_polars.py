"""Tests related to polars functionality."""
import re
from datetime import date, datetime

import polars as pl
import pytest

import patito as pt


def test_dataframe_get_method():
    """You should be able to retrieve a single row and cast to model."""

    class Product(pt.Model):
        product_id: int = pt.Field(unique=True)
        price: float

    df = pt.DataFrame({"product_id": [1, 2], "price": [9.99, 19.99]})

    # Validation does nothing, as no model is specified
    with pytest.raises(TypeError):
        df.validate()

    # But if we specify the model, it makes sense
    df.set_model(Product).validate()

    untyped_product = df.get(pl.col("product_id") == 1)
    assert untyped_product.price == 9.99

    typed_product = df.set_model(Product).get(pl.col("product_id") == 1)
    assert typed_product.price == 9.99

    with pytest.raises(
        pt.exceptions.MultipleRowsReturned,
        match=re.escape(r"DataFrame.get() yielded 2 rows."),
    ):
        df.get(pl.col("product_id") < 3)

    with pytest.raises(
        pt.exceptions.RowDoesNotExist,
        match=re.escape(r"DataFrame.get() yielded 0 rows."),
    ):
        df.get(pl.col("product_id") < 0)

    df.filter(pl.col("product_id") == 1).get()


def test_dataframe_set_model_method():
    """You should be able to set the associated model of a dataframe."""

    class MyModel(pt.Model):
        pass

    modelled_df = pt.DataFrame().set_model(MyModel)
    assert modelled_df.model is MyModel
    assert MyModel.DataFrame.model is MyModel


def test_fill_nan_with_defaults():
    """You should be able to fill missing values with declared defaults."""

    class DefaultModel(pt.Model):
        foo: int = 2
        bar: str = "default"

    missing_df = pt.DataFrame({"foo": [1, None], "bar": [None, "provided"]})
    filled_df = missing_df.set_model(DefaultModel).fill_null(strategy="defaults")
    correct_filled_df = pt.DataFrame({"foo": [1, 2], "bar": ["default", "provided"]})
    assert filled_df.frame_equal(correct_filled_df)


def test_preservation_of_model():
    """The model should be preserved on data frames after method invocations."""

    class DummyModel(pt.Model):
        a: int

    class AnotherDummyModel(pt.Model):
        a: int

    df_with_model = pt.DataFrame().set_model(DummyModel)

    # First many eagerly executed method calls
    assert (
        df_with_model.with_columns(pl.lit(1).alias("a"))
        .filter(pl.lit(1) == 1)
        .select(pl.col("a"))
        .model
    ) is DummyModel

    # A round-trip to lazy and back should also preserve the model
    assert df_with_model.lazy().collect().model is DummyModel

    # Since DataFrame.set_model does some trickery with self.__class__
    # it is important to test that this does not leak between different
    # sub-types of DataFrame
    df_with_another_model = pt.DataFrame().set_model(AnotherDummyModel)
    assert df_with_model.model is DummyModel
    assert df_with_another_model.model is AnotherDummyModel

    # The same goes for lazy round-trips
    assert df_with_model.lazy().collect().model is DummyModel
    assert df_with_another_model.lazy().collect().model is AnotherDummyModel

    # Round-trips for DataFrames and LazyFrames should work without models as well
    assert type(pt.DataFrame().lazy().collect()) is pt.DataFrame


def test_dataframe_model_dtype_casting():
    """You should be able to cast columns according to model type annotations."""

    class DTypeModel(pt.Model):
        implicit_int_1: int
        implicit_int_2: int
        explicit_uint: int = pt.Field(dtype=pl.UInt64)
        implicit_date: date
        implicit_datetime: datetime

    original_df = DTypeModel.DataFrame().with_columns(
        [
            # This float will be casted to an integer, and since no specific integer
            # dtype is specified, the default pl.Int64 will be used.
            pl.lit(1.0).cast(pl.Float64).alias("implicit_int_1"),
            # UInt32 is compatible with the "int" annotation, and since no explicit
            # dtype is specified, it will not be casted to the default pl.Int64
            pl.lit(1).cast(pl.UInt32).alias("implicit_int_2"),
            pl.lit(1.0).cast(pl.Float64).alias("explicit_uint"),
            # The integer will be casted to datetime 1970-01-01 00:00:00
            pl.lit(0).cast(pl.Int64).alias("implicit_date"),
            # The integer will be casted to date 1970-01-01
            pl.lit(0).cast(pl.Int64).alias("implicit_datetime"),
            # Columns not specified in the model should be left as-is
            pl.lit(True),
        ]
    )
    casted_df = original_df.cast()
    assert casted_df.dtypes == [
        pl.Int64,
        pl.UInt32,
        pl.UInt64,
        pl.Date,
        pl.Datetime,
        pl.Boolean,
    ]

    strictly_casted_df = original_df.cast(strict=True)
    assert strictly_casted_df.dtypes == [
        pl.Int64,
        pl.Int64,
        pl.UInt64,
        pl.Date,
        pl.Datetime,
        pl.Boolean,
    ]


@pytest.mark.xfail(strict=True)
def test_correct_columns_and_dtype_on_read(tmp_path):
    """A model DataFrame should aid CSV reading with column names and dtypes."""

    class Foo(pt.Model):
        a: str = pt.Field(derived_from="column_1")
        b: int = pt.Field(derived_from="column_2")

    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("1,2")

    regular_df = pl.read_csv(csv_path, has_header=False)
    assert regular_df.columns == ["column_1", "column_2"]
    assert regular_df.dtypes == [pl.Int64, pl.Int64]

    model_df = Foo.DataFrame.read_csv(csv_path, has_header=False)
    assert model_df.columns == ["a", "b"]
    assert model_df.dtypes == [pl.Utf8, pl.Int64]

    csv_path.write_text("b,a\n1,2")
    colum_specified_df = Foo.DataFrame.read_csv(csv_path, has_header=True)
    assert colum_specified_df.schema == {"b": pl.Int64, "a": pl.Utf8}

    dtype_specified_df = Foo.DataFrame.read_csv(
        csv_path, has_header=True, dtypes=[pl.Float64, pl.Float64]
    )
    assert dtype_specified_df.columns == ["b", "a"]
    assert dtype_specified_df.dtypes == [pl.Float64, pl.Float64]

    csv_path.write_text("1,2,3.1")
    unspecified_column_df = Foo.DataFrame.read_csv(csv_path, has_header=False)
    assert unspecified_column_df.columns == ["a", "b", "column_3"]
    assert unspecified_column_df.dtypes == [pl.Utf8, pl.Int64, pl.Float64]

    class DerivedModel(pt.Model):
        cents: int = pt.Field(derived_from=100 * pl.col("dollars"))

    csv_path.write_text("month,dollars\n1,2.99")
    derived_df = DerivedModel.DataFrame.read_csv(csv_path)
    assert derived_df.frame_equal(
        DerivedModel.DataFrame({"month": [1], "dollars": [2.99], "cents": [299]})
    )


def test_derive_functionality():
    """Test of Field(derived_from=...) and DataFrame.derive()."""

    class DerivedModel(pt.Model):
        underived: int
        const_derived: int = pt.Field(derived_from=pl.lit(3))
        column_derived: int = pt.Field(derived_from="underived")
        expr_derived: int = pt.Field(derived_from=2 * pl.col("underived"))
        second_order_derived: int = pt.Field(derived_from=2 * pl.col("expr_derived"))

    df = DerivedModel.DataFrame({"underived": [1, 2]})
    assert df.columns == ["underived"]
    derived_df = df.derive()
    correct_derived_df = DerivedModel.DataFrame(
        {
            "underived": [1, 2],
            "const_derived": [3, 3],
            "column_derived": [1, 2],
            "expr_derived": [2, 4],
            "second_order_derived": [4, 8],
        }
    )
    assert derived_df.frame_equal(correct_derived_df)

    # Non-compatible derive_from arguments should raise TypeError
    class InvalidModel(pt.Model):
        incompatible: int = pt.Field(derived_from=object)

    with pytest.raises(
        TypeError,
        match=r"Can not derive dataframe column from type \<class 'type'\>\.",
    ):
        InvalidModel.DataFrame().derive()


def test_recursive_derive():
    """Data.Frame.derive() infers proper derivation order and executes it, then returns columns in the order given by the model."""

    class DerivedModel(pt.Model):
        underived: int
        const_derived: int = pt.Field(derived_from=pl.lit(3))
        second_order_derived: int = pt.Field(
            derived_from=2 * pl.col("expr_derived")
        )  # requires expr_derived to be derived first
        column_derived: int = pt.Field(derived_from="underived")
        expr_derived: int = pt.Field(derived_from=2 * pl.col("underived"))

    df = DerivedModel.DataFrame({"underived": [1, 2]})
    assert df.columns == ["underived"]
    derived_df = df.derive()

    correct_derived_df = DerivedModel.DataFrame(
        {
            "underived": [1, 2],
            "const_derived": [3, 3],
            "second_order_derived": [4, 8],
            "column_derived": [1, 2],
            "expr_derived": [
                2,
                4,
            ],  # derived before second_order_derived, but remains in last position in output df according to the model
        }
    )
    assert derived_df.frame_equal(correct_derived_df)


def test_drop_method():
    """We should be able to drop columns not specified by the data frame model."""

    class Model(pt.Model):
        column_1: int

    df = Model.DataFrame({"column_1": [1, 2], "column_2": [3, 4]})

    # Originally we have all the columns
    assert df.columns == ["column_1", "column_2"]

    # If no argument is provided to drop, all columns not mentioned in the model are
    # dropped.
    assert df.drop().columns == ["column_1"]

    # We can still specify a different subset
    assert df.drop("column_1").columns == ["column_2"]

    # Or a list of columns
    assert df.drop(["column_1", "column_2"]).columns == []
