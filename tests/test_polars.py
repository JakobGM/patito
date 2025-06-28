"""Tests related to polars functionality."""

import re
from datetime import date, datetime
from io import StringIO
from typing import Optional

import polars as pl
import pytest
from pydantic import (
    AliasChoices,
    AliasGenerator,
    AliasPath,
    ConfigDict,
    ValidationError,
)

import patito as pt
from tests.examples import SmallModel


def test_dataframe_get_method() -> None:
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


def test_dataframe_set_model_method() -> None:
    """You should be able to set the associated model of a dataframe."""

    class MyModel(pt.Model):
        pass

    modelled_df = pt.DataFrame().set_model(MyModel)
    assert modelled_df.model is MyModel
    assert MyModel.DataFrame.model is MyModel


def test_lazyframe_from_existing() -> None:
    """You should be able convert a polars lazyframe to a patito LazyFrame."""

    class Model(pt.Model):
        a: int

    df = Model.LazyFrame.from_existing(pl.LazyFrame({"a": [0, 1]}))
    assert "ModelLazyFrame" in str(type(df))

    df = pt.LazyFrame.from_existing(pl.LazyFrame())
    assert isinstance(df, pt.LazyFrame)


def test_fill_nan_with_defaults() -> None:
    """You should be able to fill missing values with declared defaults."""

    class DefaultModel(pt.Model):
        foo: int = 2
        bar: str = "default"

    missing_df = pt.DataFrame({"foo": [1, None], "bar": [None, "provided"]})
    filled_df = missing_df.set_model(DefaultModel).fill_null(strategy="defaults")
    correct_filled_df = pt.DataFrame({"foo": [1, 2], "bar": ["default", "provided"]})
    assert filled_df.equals(correct_filled_df)


def test_create_missing_columns_with_defaults() -> None:
    """Columns that have default values should be created if they are missing."""

    class NestedModel(pt.Model):
        foo: int = 2
        small_model: Optional[SmallModel] = None

    class DefaultModel(pt.Model):
        foo: int = 2
        bar: Optional[str] = "default"
        small_model: Optional[SmallModel] = None  # works ok on polars==0.20.3
        nested_model: Optional[NestedModel] = None  # fails to convert on polars==0.20.3

    missing_df = pt.DataFrame({"foo": [1, 2]})
    filled_df = missing_df.set_model(DefaultModel).fill_null(strategy="defaults")
    correct_filled_df = pl.DataFrame(
        {
            "foo": [1, 2],
            "bar": ["default", "default"],
            "small_model": [None, None],
            "nested_model": [None, None],
        },
        schema=DefaultModel.dtypes,
    )
    assert filled_df.equals(correct_filled_df)


def test_create_missing_columns_with_dtype() -> None:
    """Ensure optional columns are created by model."""

    class DefaultModel(pt.Model):
        foo: int
        bar: Optional[int] = None

    missing_df = pt.DataFrame({"foo": [1, 2]})
    filled_df = missing_df.set_model(DefaultModel).fill_null(strategy="defaults")
    assert "bar" in filled_df.columns
    assert filled_df["bar"].dtype == pl.Int64


def test_preservation_of_model() -> None:
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


def test_dataframe_model_dtype_casting() -> None:
    """You should be able to cast columns according to model type annotations."""

    class DTypeModel(pt.Model):
        implicit_int: int
        explicit_uint: int = pt.Field(dtype=pl.UInt64)
        implicit_date: date
        implicit_datetime: datetime

    original_df = DTypeModel.DataFrame().with_columns(
        [
            # UInt32 is compatible with the "int" annotation, and since no explicit
            # dtype is specified, it will not be casted to the default pl.Int64
            pl.lit(1).cast(pl.UInt32).alias("implicit_int"),
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
        pl.UInt32,
        pl.Date,
        pl.Datetime,
        pl.Boolean,
    ]

    strictly_casted_df = original_df.cast(strict=True)
    assert strictly_casted_df.dtypes == [
        pl.Int64,
        pl.Date,
        pl.Datetime,
        pl.Boolean,
    ]

    some_columns_df = original_df.cast(
        strict=True, columns=["implicit_int", "implicit_date"]
    )
    assert some_columns_df.dtypes == [
        pl.Int64,
        pl.Date,
        pl.Int64,  # not casted
        pl.Boolean,
    ]


def test_correct_columns_and_dtype_on_read_regular_inferred(tmp_path):
    """The `polars.read_csv` function should infer dtypes."""
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("1,2")

    regular_df = pl.read_csv(csv_path, has_header=False)
    assert regular_df.columns == ["column_1", "column_2"]
    assert regular_df.dtypes == [pl.Int64, pl.Int64]


def test_correct_columns_and_dtype_on_read_model_dtypes(tmp_path):
    """A model DataFrame should read headerless CSVs with column names and dtypes."""

    class Foo(pt.Model):
        a: str = pt.Field()
        b: int = pt.Field()

    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("1,2")
    model_df = Foo.DataFrame.read_csv(csv_path, has_header=False)
    assert model_df.columns == ["a", "b"]
    assert model_df.dtypes == [pl.String, pl.Int64]


def test_correct_columns_and_dtype_on_read_ordered(tmp_path):
    """A model DataFrame should read headered CSVs with column names and dtypes."""

    class Foo(pt.Model):
        a: str = pt.Field()
        b: int = pt.Field()

    csv_path = tmp_path / "foo.csv"

    # in model field order
    csv_path.write_text("a,b\n1,2")
    column_specified_df_ab = Foo.DataFrame.read_csv(csv_path, has_header=True)
    assert column_specified_df_ab.schema == {"a": pl.String, "b": pl.Int64}
    assert column_specified_df_ab["a"].to_list() == ["1"]
    assert column_specified_df_ab["b"].to_list() == [2]

    # and out of order
    csv_path.write_text("b,a\n1,2")
    column_specified_df_ba = Foo.DataFrame.read_csv(csv_path, has_header=True)
    assert column_specified_df_ba.schema == {
        "a": pl.String,
        "b": pl.Int64,
    }
    assert column_specified_df_ba["a"].to_list() == ["2"]
    assert column_specified_df_ba["b"].to_list() == [1]


def test_correct_columns_and_dtype_on_read_ba_float_dtype_override(tmp_path):
    """A model DataFrame should aid CSV reading with column names and dtypes."""

    class Foo(pt.Model):
        a: str = pt.Field()
        b: int = pt.Field()

    csv_path = tmp_path / "foo.csv"
    # in fkield order
    csv_path.write_text("a,b\n1,2")
    dtype_specified_df = Foo.DataFrame.read_csv(
        csv_path, has_header=True, schema_overrides=[pl.Float64, pl.Float64]
    )
    assert dtype_specified_df.columns == ["a", "b"]
    assert dtype_specified_df.dtypes == [pl.Float64, pl.Float64]
    assert dtype_specified_df.schema == {"a": pl.Float64, "b": pl.Float64}
    assert dtype_specified_df["a"].to_list() == [1.0]
    assert dtype_specified_df["b"].to_list() == [2.0]

    # and reverse order
    csv_path.write_text("b,a\n1,2")
    dtype_specified_df = Foo.DataFrame.read_csv(
        csv_path, has_header=True, schema_overrides=[pl.Float64, pl.Float64]
    )
    assert dtype_specified_df.columns == ["a", "b"]
    assert dtype_specified_df.dtypes == [pl.Float64, pl.Float64]
    assert dtype_specified_df.schema == {"a": pl.Float64, "b": pl.Float64}
    assert dtype_specified_df["a"].to_list() == [2.0]
    assert dtype_specified_df["b"].to_list() == [1.0]


def test_correct_columns_and_dtype_on_read_third_float_col(tmp_path):
    """A model DataFrame should aid CSV reading with column names and dtypes."""

    class Foo(pt.Model):
        a: str = pt.Field()
        b: int = pt.Field()

    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("1,2,3.1")
    unspecified_column_df = Foo.DataFrame.read_csv(csv_path, has_header=False)
    assert unspecified_column_df.columns == ["a", "b", "column_3"]
    assert unspecified_column_df.dtypes == [pl.String, pl.Int64, pl.Float64]


def test_correct_columns_and_dtype_on_read_derived(tmp_path):
    """A model DataFrame should aid CSV reading with column names and dtypes."""
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("month,dollars\n1,2.99")

    class DerivedModel(pt.Model):
        month: int = pt.Field()
        dollars: float = pt.Field()
        cents: int = pt.Field(derived_from=100 * pl.col("dollars"))

    derived_df = DerivedModel.DataFrame.read_csv(csv_path)
    assert derived_df.columns == ["month", "dollars", "cents"]
    assert derived_df.equals(
        DerivedModel.DataFrame({"month": [1], "dollars": [2.99], "cents": [299]})
    )


def test_correct_columns_and_dtype_on_read_alias_gen(tmp_path):
    """A model DataFrame should apply aliases to CSV columns."""
    csv_path = tmp_path / "foo.csv"
    csv_path.write_text("a,b\n1,2")

    class AliasedModel(pt.Model):
        model_config = ConfigDict(
            alias_generator=AliasGenerator(validation_alias=str.upper)
        )

        A: int = pt.Field()
        B: int = pt.Field()

    aliased_df = AliasedModel.DataFrame.read_csv(csv_path)
    assert aliased_df.columns == ["A", "B"]
    assert aliased_df.equals(AliasedModel.DataFrame({"A": [1], "B": [2]}))


def test_derive_functionality() -> None:
    """Test of Field(derived_from=...) and DataFrame.derive()."""

    class DerivedModel(pt.Model):
        underived: int
        const_derived: int = pt.Field(derived_from=pl.lit(3))
        column_derived: int = pt.Field(derived_from="underived")
        expr_derived: int = pt.Field(derived_from=2 * pl.col("underived"))
        second_order_derived: int = pt.Field(derived_from=2 * pl.col("expr_derived"))

    assert DerivedModel.derived_columns == {
        "const_derived",
        "column_derived",
        "expr_derived",
        "second_order_derived",
    }

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
    assert derived_df.equals(correct_derived_df)

    # Non-compatible derive_from arguments should raise TypeError
    with pytest.raises(ValidationError):

        class InvalidModel(pt.Model):
            incompatible: int = pt.Field(derived_from=object)


def test_recursive_derive() -> None:
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
    assert derived_df.equals(correct_derived_df)


def test_derive_subset() -> None:
    """Test derived columns."""

    class DerivedModel(pt.Model):
        underived: int
        derived: Optional[int] = pt.Field(default=None, derived_from="underived")
        expr_derived: int = pt.Field(
            derived_from=2 * pl.col("derived")
        )  # depends on derived

    df = DerivedModel.DataFrame({"underived": [1, 2]})
    correct_derived_df = DerivedModel.DataFrame(
        {
            "underived": [1, 2],
            "expr_derived": [2, 4],
        }
    )
    assert df.derive(
        columns=["expr_derived"]
    ).equals(
        correct_derived_df
    )  # only include "expr_derived" in output, but ensure that "derived" was derived recursively


def test_derive_on_defaults() -> None:
    """Test derive with default values."""

    class DerivedModel(pt.Model):
        underived: int
        derived: Optional[int] = pt.Field(default=None, derived_from="underived")

    df = DerivedModel.DataFrame([DerivedModel(underived=1), DerivedModel(underived=2)])
    derived_df = df.derive()

    correct_derived_df = DerivedModel.DataFrame(
        {
            "underived": [1, 2],
            "derived": [1, 2],
        }
    )
    assert derived_df.equals(correct_derived_df)


def test_lazy_derive() -> None:
    """Test derive with LazyFrame."""

    class DerivedModel(pt.Model):
        underived: int
        derived: Optional[int] = pt.Field(default=None, derived_from="underived")

    ldf = DerivedModel.LazyFrame({"underived": [1, 2]})
    assert ldf.collect_schema().names() == ["underived"]
    derived_ldf = ldf.derive()
    assert derived_ldf.collect_schema().names() == ["underived", "derived"]
    df = derived_ldf.collect()

    correct_derived_df = DerivedModel.DataFrame(
        {
            "underived": [1, 2],
            "derived": [1, 2],
        }
    )
    assert df.equals(correct_derived_df)


def test_drop_method() -> None:
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


def test_polars_conversion():
    """You should be able to convert a DataFrame to a polars DataFrame."""

    class Model(pt.Model):
        a: int
        b: str

    df = Model.DataFrame({"a": [1, 2], "b": ["foo", "bar"]})
    polars_df = df.as_polars()
    assert isinstance(polars_df, pl.DataFrame)
    assert not isinstance(polars_df, pt.DataFrame)
    assert polars_df.shape == (2, 2)
    assert polars_df.columns == ["a", "b"]
    assert polars_df.dtypes == [pl.Int64, pl.String]


def test_validation_alias() -> None:
    """Ensure validation_alias allows multiple column names to be parsed for one field."""

    class AliasModel(pt.Model):
        my_val_a: int = pt.Field(validation_alias="myValA")
        my_val_b: int = pt.Field(
            validation_alias=AliasChoices("my_val_b", "myValB", "myValB2")
        )
        my_val_c: int
        first_name: str = pt.Field(validation_alias=AliasPath("names", 0))
        last_name: str = pt.Field(
            validation_alias=AliasChoices("lastName", AliasPath("names", 1))
        )

    examples = [
        {"myValA": 1, "myValB": 1, "my_val_c": 1, "names": ["fname1", "lname1"]},
        {"myValA": 2, "myValB": 2, "my_val_c": 2, "names": ["fname2", "lname2"]},
        {
            "my_val_a": 3,
            "myValB2": 3,
            "my_val_c": 3,
            "names": ["fname3"],
            "last_name": "lname3",
        },
        {
            "my_val_a": 4,
            "my_val_b": 4,
            "my_val_c": 4,
            "first_name": "fname4",
            "last_name": "lname4",
        },
    ]

    # check record with all aliases
    df = (
        AliasModel.LazyFrame([examples[0]])
        .unalias()
        .cast(strict=True)
        .collect()
        .validate()
    )
    assert df.columns == AliasModel.columns

    # check record with no aliases
    df = (
        AliasModel.LazyFrame([examples[3]])
        .unalias()
        .cast(strict=True)
        .collect()
        .validate()
    )
    assert df.columns == AliasModel.columns

    # check records with mixed aliases
    df = AliasModel.LazyFrame(examples).unalias().cast(strict=True).collect().validate()
    assert df.columns == AliasModel.columns


def test_validation_returns_df() -> None:
    """Ensure DataFrame.validate() returns a DataFrame."""

    class Model(pt.Model):
        a: int

    df = Model.DataFrame({"a": [1, 2]})
    assert df.validate().equals(df)


def test_alias_generator_read_csv() -> None:
    """Ensure validation alias is applied to read_csv."""

    class AliasGeneratorModel(pt.Model):
        model_config = ConfigDict(
            alias_generator=AliasGenerator(validation_alias=str.title),
        )

        My_Val_A: int
        My_Val_B: Optional[int] = None

    csv_data = StringIO("my_val_a,my_val_b\n1,")
    df = AliasGeneratorModel.DataFrame.read_csv(csv_data)
    df.validate()
    assert df.to_dicts() == [{"My_Val_A": 1, "My_Val_B": None}]


def test_iter_models() -> None:
    """Ensure iter_models() returns a generator of models."""

    class Model(pt.Model):
        a: int

    # Test with extra column to ensure column is dropped before validation
    df = Model.DataFrame({"a": [1, 2], "b": [3, 4]})
    models = df.iter_models()
    m1 = next(models)
    m2 = next(models)
    with pytest.raises(StopIteration):
        next(models)

    assert isinstance(m1, Model)
    assert isinstance(m2, Model)
    assert m1.a == 1
    assert m2.a == 2


def test_iter_models_to_list() -> None:
    """Ensure to_list() returns a list of models."""

    class Model(pt.Model):
        a: int

    df = Model.DataFrame({"a": [1, 2]})
    models = df.iter_models().to_list()
    assert models[0].a == 1
    assert models[1].a == 2
    for model in models:
        assert isinstance(model, Model)
