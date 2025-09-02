"""Logic related to the wrapping of the polars data frame library."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
)

import polars as pl
from polars._typing import IntoExpr
from pydantic import AliasChoices, AliasPath, create_model

from patito._pydantic.column_info import ColumnInfo
from patito.exceptions import MultipleRowsReturned, RowDoesNotExist

if TYPE_CHECKING:
    import numpy as np

    from patito.pydantic import Model

DF = TypeVar("DF", bound="DataFrame")
LDF = TypeVar("LDF", bound="LazyFrame")
ModelType = TypeVar("ModelType", bound="Model")
OtherModelType = TypeVar("OtherModelType", bound="Model")
T = TypeVar("T")


class ModelGenerator(Iterator[ModelType], Generic[ModelType]):
    """An iterator that can be converted to a list."""

    def __init__(self, iterator: Iterator[ModelType]) -> None:
        """Construct a ModelGenerator from an iterator."""
        self._iterator = iterator

    def to_list(self) -> list[ModelType]:
        """Convert iterator to list."""
        return list(self)

    def __next__(self) -> ModelType:  # noqa: D105
        return next(self._iterator)

    def __iter__(self) -> Iterator[ModelType]:  # noqa: D105
        return self


class LazyFrame(pl.LazyFrame, Generic[ModelType]):
    """LazyFrame class associated to DataFrame."""

    model: type[ModelType]

    def set_model(self, model: type[OtherModelType]) -> LazyFrame[OtherModelType]:
        """Associate a given patito ``Model`` with the dataframe.

        The model schema is used by methods that depend on a model being associated with
        the given dataframe such as :ref:`DataFrame.validate() <DataFrame.validate>`
        and :ref:`DataFrame.get() <DataFrame.get>`.

        ``DataFrame(...).set_model(Model)`` is equivalent with ``Model.DataFrame(...)``.

        Args:
            model (Model): Sub-class of ``patito.Model`` declaring the schema of the
                dataframe.

        Returns:
            DataFrame[Model]: Returns the same dataframe, but with an attached model
            that is required for certain model-specific dataframe methods to work.

        Examples:
            >>> from typing_extensions import Literal
            >>> import patito as pt
            >>> import polars as pl
            >>> class SchoolClass(pt.Model):
            ...     year: int = pt.Field(dtype=pl.UInt16)
            ...     letter: Literal["A", "B"] = pt.Field(dtype=pl.Categorical)
            ...
            >>> classes = pt.DataFrame(
            ...     {"year": [1, 1, 2, 2], "letter": list("ABAB")}
            ... ).set_model(SchoolClass)
            >>> classes
            shape: (4, 2)
            ┌──────┬────────┐
            │ year ┆ letter │
            │ ---  ┆ ---    │
            │ i64  ┆ str    │
            ╞══════╪════════╡
            │ 1    ┆ A      │
            │ 1    ┆ B      │
            │ 2    ┆ A      │
            │ 2    ┆ B      │
            └──────┴────────┘
            >>> casted_classes = classes.cast()
            >>> casted_classes
            shape: (4, 2)
            ┌──────┬────────┐
            │ year ┆ letter │
            │ ---  ┆ ---    │
            │ u16  ┆ cat    │
            ╞══════╪════════╡
            │ 1    ┆ A      │
            │ 1    ┆ B      │
            │ 2    ┆ A      │
            │ 2    ┆ B      │
            └──────┴────────┘
            >>> casted_classes.validate()

        """
        return model.LazyFrame._from_pyldf(self._ldf)  # type: ignore

    def collect(
        self,
        *args,
        **kwargs,
    ) -> DataFrame[ModelType]:  # noqa: DAR101, DAR201
        """Collect into a DataFrame.

        See documentation of polars.DataFrame.collect for full description of
        parameters.
        """
        background = kwargs.pop("background", False)
        df: pl.DataFrame = super().collect(*args, background=background, **kwargs)
        df = DataFrame(df)
        if getattr(self, "model", False):
            df = df.set_model(self.model)
        return df

    def derive(self: LDF, columns: list[str] | None = None) -> LDF:
        """Populate columns which have ``pt.Field(derived_from=...)`` definitions.

        If a column field on the data frame model has ``patito.Field(derived_from=...)``
        specified, the given value will be used to define the column. If
        ``derived_from`` is set to a string, the column will be derived from the given
        column name. Alternatively, an arbitrary polars expression can be given, the
        result of which will be used to populate the column values.

        Args:
            columns: Optionally, a list of column names to derive. If not provided, all
                columns are used.

        Returns:
            DataFrame[Model]: A new dataframe where all derivable columns are provided.

        Raises:
            TypeError: If the ``derived_from`` parameter of ``patito.Field`` is given
                as something else than a string or polars expression.

        Examples:
            >>> import patito as pt
            >>> import polars as pl
            >>> class Foo(pt.Model):
            ...     bar: int = pt.Field(derived_from="foo")
            ...     double_bar: int = pt.Field(derived_from=2 * pl.col("bar"))
            ...
            >>> Foo.DataFrame({"foo": [1, 2]}).derive()
            shape: (2, 3)
            ┌─────┬────────────┬─────┐
            │ bar ┆ double_bar ┆ foo │
            │ --- ┆ ---        ┆ --- │
            │ i64 ┆ i64        ┆ i64 │
            ╞═════╪════════════╪═════╡
            │ 1   ┆ 2          ┆ 1   │
            │ 2   ┆ 4          ┆ 2   │
            └─────┴────────────┴─────┘

        """
        derived_columns = []
        props = self.model._schema_properties()
        original_columns = set(self.collect_schema())
        to_derive = self.model.derived_columns if columns is None else columns
        for column_name in to_derive:
            if column_name not in derived_columns:
                self, _derived_columns = self._derive_column(
                    self, column_name, self.model.column_infos
                )
                derived_columns.extend(_derived_columns)
        out_cols = [
            x for x in props if x in original_columns.union(to_derive)
        ]  # ensure that model columns are first and in the correct order
        out_cols += [
            x for x in original_columns.union(to_derive) if x not in out_cols
        ]  # collect columns originally in data frame that are not in the model and append to end of df
        return self.select(out_cols)

    def _derive_column(
        self,
        lf: LDF,
        column_name: str,
        column_infos: dict[str, ColumnInfo],
    ) -> tuple[LDF, Sequence[str]]:
        if (
            column_infos.get(column_name, None) is None
            or column_infos[column_name].derived_from is None
        ):
            return lf, []

        derived_from = column_infos[column_name].derived_from
        dtype = self.model.dtypes[column_name]
        derived_columns = []

        if isinstance(derived_from, str):
            lf = lf.with_columns(pl.col(derived_from).cast(dtype).alias(column_name))
        elif isinstance(derived_from, pl.Expr):
            root_cols = derived_from.meta.root_names()
            while root_cols:
                root_col = root_cols.pop()
                lf, _derived_columns = self._derive_column(lf, root_col, column_infos)
                derived_columns.extend(_derived_columns)
            lf = lf.with_columns(derived_from.cast(dtype).alias(column_name))
        else:
            raise TypeError(
                f"Can not derive dataframe column from type {type(derived_from)}."
            )
        derived_columns.append(column_name)
        return lf, derived_columns

    def unalias(self: LDF) -> LDF:
        """Un-aliases column names using information from pydantic validation_alias.

        In order of preference - model field name then validation_aliases in order of occurrence

        limitation - AliasChoice validation type only supports selecting a single element of an array

        Returns:
            DataFrame[Model]: A dataframe with columns normalized to model names.

        """
        if not any(fi.validation_alias for fi in self.model.model_fields.values()):
            return self
        exprs = []

        def to_expr(va: str | AliasPath | AliasChoices) -> pl.Expr | None:
            if isinstance(va, str):
                return pl.col(va) if va in self.collect_schema() else None
            elif isinstance(va, AliasPath):
                if len(va.path) != 2 or not isinstance(va.path[1], int):
                    raise NotImplementedError(
                        f"TODO figure out how this AliasPath behaves ({va})"
                    )
                return (
                    pl.col(str(va.path[0])).list.get(va.path[1], null_on_oob=True)
                    if va.path[0] in self.collect_schema()
                    else None
                )
            elif isinstance(va, AliasChoices):
                local_expr: pl.Expr | None = None
                for choice in va.choices:
                    if (part := to_expr(choice)) is not None:
                        local_expr = (
                            local_expr.fill_null(value=part)
                            if local_expr is not None
                            else part
                        )
                return local_expr
            else:
                raise NotImplementedError(
                    f"unknown validation_alias type {field_info.validation_alias}"
                )

        for name, field_info in self.model.model_fields.items():
            if field_info.validation_alias is None:
                exprs.append(pl.col(name))
            else:
                expr = to_expr(field_info.validation_alias)
                if name in self.collect_schema().names():
                    if expr is None:
                        exprs.append(pl.col(name))
                    else:
                        exprs.append(pl.col(name).fill_null(value=expr))
                elif expr is not None:
                    exprs.append(expr.alias(name))

        return self.select(exprs)

    def cast(
        self: LDF, strict: bool = False, columns: Sequence[str] | None = None
    ) -> LDF:
        """Cast columns to `dtypes` specified by the associated Patito model.

        Args:
            strict: If set to ``False``, columns which are technically compliant with
                the specified field type, will not be casted. For example, a column
                annotated with ``int`` is technically compliant with ``pl.UInt8``, even
                if ``pl.Int64`` is the default dtype associated with ``int``-annotated
                fields. If ``strict`` is set to ``True``, the resulting dtypes will
                be forced to the default dtype associated with each python type.
            columns: Optionally, a list of column names to cast. If not provided, all
                columns are casted.

        Returns:
            LazyFrame[Model]: A dataframe with columns casted to the correct dtypes.

        Examples:
            Create a simple model:

            >>> import patito as pt
            >>> import polars as pl
            >>> class Product(pt.Model):
            ...     name: str
            ...     cent_price: int = pt.Field(dtype=pl.UInt16)
            ...

            Now we can use this model to cast some simple data:

            >>> Product.LazyFrame({"name": ["apple"], "cent_price": ["8"]}).cast().collect()
            shape: (1, 2)
            ┌───────┬────────────┐
            │ name  ┆ cent_price │
            │ ---   ┆ ---        │
            │ str   ┆ u16        │
            ╞═══════╪════════════╡
            │ apple ┆ 8          │
            └───────┴────────────┘

        """
        properties = self.model._schema_properties()
        valid_dtypes = self.model.valid_dtypes
        default_dtypes = self.model.dtypes
        columns = columns or self.collect_schema().names()
        exprs = []
        for column, current_dtype in self.collect_schema().items():
            if (column not in columns) or (column not in properties):
                exprs.append(pl.col(column))
            elif "dtype" in properties[column]:
                exprs.append(pl.col(column).cast(properties[column]["dtype"]))
            elif not strict and current_dtype in valid_dtypes[column]:
                exprs.append(pl.col(column))
            else:
                exprs.append(pl.col(column).cast(default_dtypes[column]))
        return self.with_columns(exprs)

    @classmethod
    def from_existing(cls: type[LDF], lf: pl.LazyFrame) -> LDF:
        """Construct a patito.DataFrame object from an existing polars.DataFrame object."""
        if getattr(cls, "model", False):
            return cls.model.LazyFrame._from_pyldf(super().lazy(lf)._ldf)  # type: ignore

        return LazyFrame._from_pyldf(lf._ldf)  # type: ignore


class DataFrame(pl.DataFrame, Generic[ModelType]):
    """A sub-class of polars.DataFrame with additional functionality related to Model.

    Two different methods are available for constructing model-aware data frames.
    Assume a simple model with two fields:

    >>> import patito as pt
    >>> class Product(pt.Model):
    ...     name: str
    ...     price_in_cents: int
    ...

    We can construct a data frame containing products and then associate the
    :code:`Product` model to the data frame using ``DataFrame.set_model``:

    >>> df = pt.DataFrame({"name": ["apple", "banana"], "price": [25, 61]}).set_model(
    ...     Product
    ... )

    Alternatively, we can use the custom :code:`Product.DataFrame` class which
    automatically associates the :code:`Product` model to the data frame at
    instantiation.

    >>> df = Product.DataFrame({"name": ["apple", "banana"], "price": [25, 61]})

    The :code:`df` data frame now has a set of model-aware methods such as as
    :ref:`Product.validate <DataFrame.validate>`.
    """

    model: type[ModelType]

    def lazy(self: DataFrame[ModelType]) -> LazyFrame[ModelType]:
        """Convert DataFrame into LazyFrame.

        See documentation of polars.DataFrame.lazy() for full description.

        Returns:
            A new LazyFrame object.

        """
        if getattr(self, "model", False):
            return self.model.LazyFrame._from_pyldf(super().lazy()._ldf)  # type: ignore

        return LazyFrame._from_pyldf(super().lazy()._ldf)  # type: ignore

    def set_model(self, model: type[OtherModelType]) -> DataFrame[OtherModelType]:
        """Associate a given patito ``Model`` with the dataframe.

        The model schema is used by methods that depend on a model being associated with
        the given dataframe such as :ref:`DataFrame.validate() <DataFrame.validate>`
        and :ref:`DataFrame.get() <DataFrame.get>`.

        ``DataFrame(...).set_model(Model)`` is equivalent with ``Model.DataFrame(...)``.

        Args:
            model (Model): Sub-class of ``patito.Model`` declaring the schema of the
                dataframe.

        Returns:
            DataFrame[Model]: Returns the same dataframe, but with an attached model
            that is required for certain model-specific dataframe methods to work.

        Examples:
            >>> from typing_extensions import Literal
            >>> import patito as pt
            >>> import polars as pl
            >>> class SchoolClass(pt.Model):
            ...     year: int = pt.Field(dtype=pl.UInt16)
            ...     letter: Literal["A", "B"] = pt.Field(dtype=pl.Categorical)
            ...
            >>> classes = pt.DataFrame(
            ...     {"year": [1, 1, 2, 2], "letter": list("ABAB")}
            ... ).set_model(SchoolClass)
            >>> classes
            shape: (4, 2)
            ┌──────┬────────┐
            │ year ┆ letter │
            │ ---  ┆ ---    │
            │ i64  ┆ str    │
            ╞══════╪════════╡
            │ 1    ┆ A      │
            │ 1    ┆ B      │
            │ 2    ┆ A      │
            │ 2    ┆ B      │
            └──────┴────────┘
            >>> casted_classes = classes.cast()
            >>> casted_classes
            shape: (4, 2)
            ┌──────┬────────┐
            │ year ┆ letter │
            │ ---  ┆ ---    │
            │ u16  ┆ cat    │
            ╞══════╪════════╡
            │ 1    ┆ A      │
            │ 1    ┆ B      │
            │ 2    ┆ A      │
            │ 2    ┆ B      │
            └──────┴────────┘
            >>> casted_classes.validate()

        """
        return model.DataFrame._from_pydf(self._df)

    def unalias(self: DF) -> DF:
        """Un-aliases column names using information from pydantic validation_alias.

        In order of preference - model field name then validation_aliases in order of occurrence

        limitation - AliasChoice validation type only supports selecting a single element of an array

        Returns:
            DataFrame[Model]: A dataframe with columns normalized to model names.

        """
        return self.lazy().unalias().collect()

    def cast(
        self: DF, strict: bool = False, columns: Sequence[str] | None = None
    ) -> DF:
        """Cast columns to `dtypes` specified by the associated Patito model.

        Args:
            strict: If set to ``False``, columns which are technically compliant with
                the specified field type, will not be casted. For example, a column
                annotated with ``int`` is technically compliant with ``pl.UInt8``, even
                if ``pl.Int64`` is the default dtype associated with ``int``-annotated
                fields. If ``strict`` is set to ``True``, the resulting dtypes will
                be forced to the default dtype associated with each python type.
            columns: Optionally, a list of column names to cast. If not provided, all
                columns are casted.

        Returns:
            DataFrame[Model]: A dataframe with columns casted to the correct dtypes.

        Examples:
            Create a simple model:

            >>> import patito as pt
            >>> import polars as pl
            >>> class Product(pt.Model):
            ...     name: str
            ...     cent_price: int = pt.Field(dtype=pl.UInt16)
            ...

            Now we can use this model to cast some simple data:

            >>> Product.DataFrame({"name": ["apple"], "cent_price": ["8"]}).cast()
            shape: (1, 2)
            ┌───────┬────────────┐
            │ name  ┆ cent_price │
            │ ---   ┆ ---        │
            │ str   ┆ u16        │
            ╞═══════╪════════════╡
            │ apple ┆ 8          │
            └───────┴────────────┘

        """
        return self.lazy().cast(strict=strict, columns=columns).collect()

    def drop(
        self: DF,
        columns: str | Collection[str] | None = None,
    ) -> DF:
        """Drop one or more columns from the dataframe.

        If ``name`` is not provided then all columns `not` specified by the associated
        patito model, for instance set with
        :ref:`DataFrame.set_model <DataFrame.set_model>`, are dropped.

        Args:
            columns: A single column string name, or list of strings, indicating
                which columns to drop. If not specified, all columns *not*
                specified by the associated dataframe model will be dropped.

        Returns:
            DataFrame[Model]: New dataframe without the specified columns.

        Examples:
            >>> import patito as pt
            >>> class Model(pt.Model):
            ...     column_1: int
            ...
            >>> Model.DataFrame({"column_1": [1, 2], "column_2": [3, 4]}).drop()
            shape: (2, 1)
            ┌──────────┐
            │ column_1 │
            │ ---      │
            │ i64      │
            ╞══════════╡
            │ 1        │
            │ 2        │
            └──────────┘

        """
        if columns is not None:
            # Use super() to call polars DataFrame drop method directly
            return self._from_pydf(super().drop(columns)._df)
        else:
            return self.drop(list(set(self.columns) - set(self.model.columns)))

    def validate(self, columns: Sequence[str] | None = None, **kwargs: Any):
        """Validate the schema and content of the dataframe.

        You must invoke ``.set_model()`` before invoking ``.validate()`` in order
        to specify how the dataframe should be validated.

        Returns:
            DataFrame[Model]: The original patito dataframe, if correctly validated.

        Raises:
            patito.exceptions.DataFrameValidationError: If the dataframe does not match the
                specified schema.

            TypeError: If ``DataFrame.set_model()`` has not been invoked prior to
                validation. Note that ``patito.Model.DataFrame`` automatically invokes
                ``DataFrame.set_model()`` for you.

        Examples:
            >>> import patito as pt


            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     temperature_zone: Literal["dry", "cold", "frozen"]
            ...     is_for_sale: bool
            ...

            >>> df = pt.DataFrame(
            ...     {
            ...         "product_id": [1, 1, 3],
            ...         "temperature_zone": ["dry", "dry", "oven"],
            ...     }
            ... ).set_model(Product)
            >>> try:
            ...     df.validate()
            ... except pt.DataFrameValidationError as exc:
            ...     print(exc)
            ...
            3 validation errors for Product
            is_for_sale
              Missing column (type=type_error.missingcolumns)
            product_id
              2 rows with duplicated values. (type=value_error.rowvalue)
            temperature_zone
              Rows with invalid values: {'oven'}. (type=value_error.rowvalue)

        """
        if not hasattr(self, "model"):
            raise TypeError(
                f"You must invoke {self.__class__.__name__}.set_model() "
                f"before invoking {self.__class__.__name__}.validate()."
            )
        self.model.validate(dataframe=self, columns=columns, **kwargs)
        return self

    def derive(self: DF, columns: list[str] | None = None) -> DF:
        """Populate columns which have ``pt.Field(derived_from=...)`` definitions.

        If a column field on the data frame model has ``patito.Field(derived_from=...)``
        specified, the given value will be used to define the column. If
        ``derived_from`` is set to a string, the column will be derived from the given
        column name. Alternatively, an arbitrary polars expression can be given, the
        result of which will be used to populate the column values.

        Returns:
            DataFrame[Model]: A new dataframe where all derivable columns are provided.

        Raises:
            TypeError: If the ``derived_from`` parameter of ``patito.Field`` is given
                as something else than a string or polars expression.

        Examples:
            >>> import patito as pt
            >>> import polars as pl
            >>> class Foo(pt.Model):
            ...     bar: int = pt.Field(derived_from="foo")
            ...     double_bar: int = pt.Field(derived_from=2 * pl.col("bar"))
            ...
            >>> Foo.DataFrame({"foo": [1, 2]}).derive()
            shape: (2, 3)
            ┌─────┬────────────┬─────┐
            │ bar ┆ double_bar ┆ foo │
            │ --- ┆ ---        ┆ --- │
            │ i64 ┆ i64        ┆ i64 │
            ╞═════╪════════════╪═════╡
            │ 1   ┆ 2          ┆ 1   │
            │ 2   ┆ 4          ┆ 2   │
            └─────┴────────────┴─────┘

        """
        return cast(DF, self.lazy().derive(columns=columns).collect())

    def fill_null(
        self: DF,
        value: Any | None = None,
        strategy: Literal[
            "forward", "backward", "min", "max", "mean", "zero", "one", "defaults"
        ]
        | None = None,
        limit: int | None = None,
        matches_supertype: bool = True,
    ) -> DF:
        """Fill null values using a filling strategy, literal, or ``Expr``.

        If ``"defaults"`` is provided as the strategy, the model fields with default
        values are used to fill missing values.

        Args:
            value: Value used to fill null values.
            strategy: Accepts the same arguments as ``polars.DataFrame.fill_null`` in
                addition to ``"defaults"`` which will use the field's default value if
                provided.
            limit: The number of consecutive null values to forward/backward fill.
                Only valid if ``strategy`` is ``"forward"`` or ``"backward"``.
            matches_supertype: Fill all matching supertype of the fill ``value``.


        Returns:
            DataFrame[Model]: A new dataframe with nulls filled in according to the
            provided ``strategy`` parameter.

        Example:
            >>> import patito as pt
            >>> class Product(pt.Model):
            ...     name: str
            ...     price: int = 19
            ...
            >>> df = Product.DataFrame(
            ...     {"name": ["apple", "banana"], "price": [10, None]}
            ... )
            >>> df.fill_null(strategy="defaults")
            shape: (2, 2)
            ┌────────┬───────┐
            │ name   ┆ price │
            │ ---    ┆ ---   │
            │ str    ┆ i64   │
            ╞════════╪═══════╡
            │ apple  ┆ 10    │
            │ banana ┆ 19    │
            └────────┴───────┘

        """
        if strategy != "defaults":  # pragma: no cover
            return cast(  # pyright: ignore[redundant-cast]
                DF,
                super().fill_null(
                    value=value,
                    strategy=strategy,
                    limit=limit,
                    matches_supertype=matches_supertype,
                ),
            )
        return self.with_columns(
            [
                (
                    pl.col(column).fill_null(
                        pl.lit(default_value, self.model.dtypes[column])
                    )
                    if column in self.columns
                    else pl.lit(default_value, self.model.dtypes[column]).alias(column)
                )
                for column, default_value in self.model.defaults.items()
            ]
        ).set_model(self.model)  # type: ignore

    def get(self, predicate: pl.Expr | None = None) -> ModelType:
        """Fetch the single row that matches the given polars predicate.

        If you expect a data frame to already consist of one single row,
        you can use ``.get()`` without any arguments to return that row.

        Raises:
            RowDoesNotExist: If zero rows evaluate to true for the given predicate.
            MultipleRowsReturned: If more than one row evaluates to true for the given
                predicate.
            RuntimeError: The superclass of both ``RowDoesNotExist`` and
                ``MultipleRowsReturned`` if you want to catch both exceptions with the
                same class.

        Args:
            predicate: A polars expression defining the criteria of the filter.

        Returns:
            Model: A pydantic-derived base model representing the given row.

        Example:
            >>> import patito as pt
            >>> import polars as pl
            >>> df = pt.DataFrame({"product_id": [1, 2, 3], "price": [10, 10, 20]})

            The ``.get()`` will by default return a dynamically constructed pydantic
            model if no model has been associated with the given dataframe:

            >>> df.get(pl.col("product_id") == 1)
            UntypedRow(product_id=1, price=10)

            If a Patito model has been associated with the dataframe, by the use of
            :ref:`DataFrame.set_model()<DataFrame.set_model>`, then the given model will
            be used to represent the return type:

            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     price: float
            ...
            >>> df.set_model(Product).get(pl.col("product_id") == 1)
            Product(product_id=1, price=10.0)

            You can invoke ``.get()`` without any arguments on dataframes containing
            exactly one row:

            >>> df.filter(pl.col("product_id") == 1).get()
            UntypedRow(product_id=1, price=10)

            If the given predicate matches multiple rows a ``MultipleRowsReturned`` will
            be raised:

            >>> try:
            ...     df.get(pl.col("price") == 10)
            ... except pt.exceptions.MultipleRowsReturned as e:
            ...     print(e)
            ...
            DataFrame.get() yielded 2 rows.

            If the given predicate matches zero rows a ``RowDoesNotExist`` will
            be raised:

            >>> try:
            ...     df.get(pl.col("price") == 0)
            ... except pt.exceptions.RowDoesNotExist as e:
            ...     print(e)
            ...
            DataFrame.get() yielded 0 rows.

        """
        row = self if predicate is None else self.filter(predicate)
        if row.height == 0:
            raise RowDoesNotExist(f"{self.__class__.__name__}.get() yielded 0 rows.")
        if row.height > 1:
            raise MultipleRowsReturned(
                f"{self.__class__.__name__}.get() yielded {row.height} rows."
            )

        if hasattr(self, "model"):
            return self.model.from_row(row)
        else:
            return self._pydantic_model().from_row(row)  # type: ignore

    def iter_models(
        self, validate_df: bool = True, validate_model: bool = False
    ) -> ModelGenerator[ModelType]:
        """Iterate over all rows in the dataframe as pydantic models.

        Args:
            validate_df: If set to ``True``, the dataframe will be validated before
                making models out of each row. If set to ``False``, beware that columns
                need to be the exact same as the model fields.
            validate_model: If set to ``True``, each model will be validated when
                constructing. Disabled by default since df validation should cover this case.

        Yields:
            Model: A pydantic-derived model representing the given row. .to_list() can be
                used to convert the iterator to a list.

        Raises:
            TypeError: If ``DataFrame.set_model()`` has not been invoked prior to
                iteration.

        Example:
            >>> import patito as pt
            >>> import polars as pl
            >>> class Product(pt.Model):
            ...     product_id: int = pt.Field(unique=True)
            ...     price: float

            >>> df = pt.DataFrame({"product_id": [1, 2], "price": [10., 20.]})
            >>> df = df.set_model(Product)
            >>> for product in df.iter_models():
            ...     print(product)
            ...
            Product(product_id=1, price=10.0)
            Product(product_id=2, price=20.0)

        """
        if not hasattr(self, "model"):
            raise TypeError(
                f"You must invoke {self.__class__.__name__}.set_model() "
                f"before invoking {self.__class__.__name__}.iter_models()."
            )

        df = self.validate(drop_superfluous_columns=True) if validate_df else self

        def _iter_models_with_validate(
            _df: DataFrame[ModelType],
        ) -> Iterator[ModelType]:
            for row in _df.iter_rows(named=True):
                yield self.model(**row)

        def _iter_models_without_validate(
            _df: DataFrame[ModelType],
        ) -> Iterator[ModelType]:
            for row in _df.iter_rows(named=True):
                yield self.model.model_construct(**row)

        _iter_models = (
            _iter_models_with_validate
            if validate_model
            else _iter_models_without_validate
        )
        return ModelGenerator(_iter_models(df))

    def _pydantic_model(self) -> type[Model]:
        """Dynamically construct patito model compliant with dataframe.

        Returns:
            A pydantic model class where all the rows have been specified as
                `typing.Any` fields.

        """
        from patito.pydantic import Model

        pydantic_annotations = {column: (Any, ...) for column in self.columns}
        return cast(
            type[Model],
            create_model(  # type: ignore
                "UntypedRow",
                __base__=Model,
                **pydantic_annotations,  # pyright: ignore
            ),
        )

    def as_polars(self) -> pl.DataFrame:
        """Convert patito dataframe to polars dataframe."""
        return pl.DataFrame._from_pydf(self._df)

    @classmethod
    def read_csv(  # type: ignore[no-untyped-def]
        cls: type[DF],
        *args,  # noqa: ANN002
        **kwargs,  # noqa: ANN003
    ) -> DF:
        r"""Read CSV and apply correct column name and types from model.

        If any fields have ``derived_from`` specified, the given expression will be used
        to populate the given column(s).

        Args:
            *args: All positional arguments are forwarded to ``polars.read_csv``.
            **kwargs: All keyword arguments are forwarded to ``polars.read_csv``.

        Returns:
            DataFrame[Model]: A dataframe representing the given CSV file data.

        Examples:
            The ``DataFrame.read_csv`` method can be used to automatically set the
            correct column names when reading CSV files without headers.

            >>> import io
            >>> import patito as pt
            >>> class CSVModel(pt.Model):
            ...     a: float
            ...     b: str
            ...
            >>> csv_file = io.StringIO("1,2")
            >>> CSVModel.DataFrame.read_csv(csv_file, has_header=False)
            shape: (1, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ f64 ┆ str │
            ╞═════╪═════╡
            │ 1.0 ┆ 2   │
            └─────┴─────┘

            The ``derived_from`` paramater of ``pt.Field`` allows you to specify
            the mapping between the CSV file's column names, and the final column names
            you intend to construct.

            >>> import io
            >>> import patito as pt
            >>> class CSVModel(pt.Model):
            ...     a: float
            ...     b: str = pt.Field(derived_from="source_of_b")
            ...
            >>> csv_file = io.StringIO("a,source_of_b\n1,1")


            # >>> CSVModel.DataFrame.read_csv(csv_file).drop()
            # shape: (1, 2)
            # ┌─────┬─────┐
            # │ a   ┆ b   │
            # │ --- ┆ --- │
            # │ f64 ┆ str │
            # ╞═════╪═════╡
            # │ 1.0 ┆ 1   │
            # └─────┴─────┘

        """
        kwargs.setdefault("schema_overrides", cls.model.dtypes)
        has_header = kwargs.get("has_header", True)
        if not has_header and "columns" not in kwargs:
            kwargs.setdefault("new_columns", cls.model.columns)
        alias_gen = cls.model.model_config.get("alias_generator")
        if alias_gen:
            alias_func = alias_gen.validation_alias or alias_gen.alias
        if has_header and alias_gen and alias_func:
            fields_to_cols = {
                field_name: alias_func(field_name)
                for field_name in cls.model.model_fields
            }
            kwargs["schema_overrides"] = {
                fields_to_cols.get(field, field): dtype
                for field, dtype in kwargs["schema_overrides"].items()
            }
            # TODO: other forms of alias setting like in Field
        df = cls.model.DataFrame._from_pydf(pl.read_csv(*args, **kwargs)._df)
        return df.derive()

    # --- Type annotation overrides ---
    def filter(  # noqa: D102
        self: DF,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray[Any, Any] | bool,
    ) -> DF:
        return cast(DF, super().filter(predicate))

    def select(  # noqa: D102
        self: DF,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DF:
        return cast(  # pyright: ignore[redundant-cast]
            DF, super().select(*exprs, **named_exprs)
        )

    def with_columns(  # noqa: D102
        self: DF,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> DF:
        return cast(DF, super().with_columns(*exprs, **named_exprs))
