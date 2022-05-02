"""Logic related to the wrapping of the polars data frame library."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import polars as pl
from pydantic import create_model
from typing_extensions import Literal

from patito.exceptions import MultipleRowsReturned, RowDoesNotExist

if TYPE_CHECKING:
    from patito.pydantic import Model


DF = TypeVar("DF", bound="DataFrame")
ModelType = TypeVar("ModelType", bound="Model")


class DataFrame(pl.DataFrame, Generic[ModelType]):
    """
    A custom model-aware sub-class of polars.DataFrame.

    Adds additional model-related methods such as `DataFrame.set_model()`,
    `DataFrame.validate()`, `DataFrame.derive()`, and so on.
    """

    model: ModelType

    @classmethod
    def _construct_dataframe_model_class(
        cls: Type[DF], model: Model
    ) -> DataFrame[Model]:
        """
        Return custom DataFrame sub-class where DataFrame.model is set.

        Can be used to construct a DataFrame class where
        DataFrame.set_model(model) is implicitly invoked at instantiation.

        Args:
            model: A patito model which should be used to validate the dataframe.

        Returns:
            A custom DataFrame model class where DataFrame._model has been correctly
                "hard-coded" to the given model.
        """
        new_class = type(
            f"{model.schema()['title']}DataFrame",
            (cls,),  # type: ignore
            {"model": model},
        )
        new_class._lazyframe_class = type(  # type: ignore
            f"{model.__class__.__name__}LazyFrame",
            (new_class._lazyframe_class,),  # type: ignore
            {"_dataframe_class": new_class},
        )
        return cast("DataFrame[Model]", new_class)

    def set_model(self, model):  # noqa: ANN001, ANN201
        """
        Set the model which represents the data frame schema.

        The model schema is used by methods such as `DataFrame.validate()` and
        `DataFrame.get()`.

        Args:
            model: Sub-class of patito.Model declaring the schema of the dataframe.

        Returns:
            Returns the same dataframe, but with attached model validation metadata.
        """
        cls = self._construct_dataframe_model_class(model=model)
        return cast(
            DataFrame[model],
            cls._from_pydf(self._df),  # type: ignore
        )

    def cast(self: DF, strict: bool = False) -> DF:
        """
        Cast columns to dtypes specified by the Patito model.

        Args:
            strict: If set to False, columns which are technically compliant with the
                specified field type, will not be casted. For example, a column
                annotated with int is technically compliant with pl.UInt8, even if
                pl.Int64 is the default dtype associated with such fields. If strict
                is set to True, the resulting dtypes will be forced to the default dtype
                associated with each python type.

        Returns:
            A dataframe with columns casted to the correct dtypes.
        """
        properties = self.model.schema()["properties"]
        valid_dtypes = self.model.valid_dtypes
        default_dtypes = self.model.dtypes
        columns = []
        for column, current_dtype in zip(self.columns, self.dtypes):
            if column not in properties:
                columns.append(pl.col(column))
            elif "dtype" in properties[column]:
                columns.append(pl.col(column).cast(properties[column]["dtype"]))
            elif not strict and current_dtype in valid_dtypes[column]:
                columns.append(pl.col(column))
            else:
                columns.append(pl.col(column).cast(default_dtypes[column]))
        return self.with_columns(columns)

    def drop(self: DF, name: Optional[Union[str, List[str]]] = None) -> DF:
        """
        Drop one or more columns from the dataframe.

        If `name` is not provided, all columns not specified in the DataFrame
        model, set with DataFrame.set_model(), are dropped.

        Args:
            name: A single column string name, or list of strings, indicating
                which columns to drop. If not specified, all columns *not*
                specified by the associated dataframe model will be dropped.

        Returns:
            New dataframe without the specified columns.

        Examples:
            >>> import patito as pt

            >>> class Model(pt.Model):
            ...     column_1: int

            >>> Model.DataFrame({"column_1": [1, 2], "column_2": [3, 4]}).drop()
            shape: (2, 1)
            ┌──────────┐
            │ column_1 │
            │ ---      │
            │ i64      │
            ╞══════════╡
            │ 1        │
            ├╌╌╌╌╌╌╌╌╌╌┤
            │ 2        │
            └──────────┘

        """
        if name is not None:
            return super().drop(name)
        else:
            return self.drop(list(set(self.columns) - set(self.model.columns)))

    def validate(self: DF) -> DF:
        """
        Validate the schema and content of the data frame.

        You must invoke .set_model() before invoking .validate() in order
        to specify the model schema of the data frame.

        Returns:
            The original dataframe, if correctly validated.

        Raises:
            TypeError: If `DataFrame.set_model()` has not been invoked prior to
                validation. Note that `patito.Model.DataFrame` automatically invokes
                `DataFrame.set_model()` for you.
            patito.exceptions.ValidationError:  # noqa: DAR402
                If the dataframe does not match the specified schema.
        """
        if not hasattr(self, "model"):
            raise TypeError(
                f"You must invoke {self.__class__.__name__}.set_model() "
                f"before invoking {self.__class__.__name__}.validate()."
            )
        self.model.validate(dataframe=self)
        return self

    def derive(self: DF) -> DF:
        """
        Derive columns which are derived from other columns or expressions.

        If a column field on the DataFrame model has patito.Field(derived_from=...)
        specified, the given value will be used to define the column. If `derived_from`
        is set to a string, the column will be derived from the given column name.
        Alternatively, an arbitrary polars expression can be given, the result of which
        will be used to populate the column values.

        Returns:
            A new dataframe where all derivable columns are provided.

        Raises:
            TypeError: If the `derived_from` parameter of `patito.Field` is given as
                something elso than a string or polars expression.

        Examples:
            >>> import patito as pt
            >>> import polars as pl
            ...
            >>> class Foo(pt.Model):
            ...     bar: int = pt.Field(derived_from="foo")
            ...     double_bar: int = pt.Field(derived_from=2 * pl.col("bar"))
            ...
            >>> Foo.DataFrame({"foo": [1, 2]}).derive()
            shape: (2, 3)
            ┌─────┬─────┬────────────┐
            │ foo ┆ bar ┆ double_bar │
            │ --- ┆ --- ┆ ---        │
            │ i64 ┆ i64 ┆ i64        │
            ╞═════╪═════╪════════════╡
            │ 1   ┆ 1   ┆ 2          │
            ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
            │ 2   ┆ 2   ┆ 4          │
            └─────┴─────┴────────────┘
        """
        df = self.lazy()
        for column_name, props in self.model.schema()["properties"].items():
            if "derived_from" in props:
                derived_from = props["derived_from"]
                dtype = self.model.dtypes[column_name]
                if isinstance(derived_from, str):
                    df = df.with_column(
                        pl.col(derived_from).cast(dtype).alias(column_name)
                    )
                elif isinstance(derived_from, pl.Expr):
                    df = df.with_column(derived_from.cast(dtype).alias(column_name))
                else:
                    raise TypeError(
                        "Can not derive dataframe column from type "
                        f"{type(derived_from)}."
                    )
        return df.collect()

    def fill_null(
        self: DF,
        strategy: Union[
            Literal[
                "backward",
                "forward",
                "mean",
                "min",
                "max",
                "zero",
                "one",
                "defaults",
            ],
            pl.Expr,
            Any,
        ],
    ) -> DF:
        """
        Fill null values using a filling strategy, literal, or Expr.

        If "default" is provided as the strategy, the model fields with default values
        are used to fill missing values.

        Args:
            strategy: Accepts the same arguments as `polars.DataFrame.fill_null` in
                addition to `"defaults"` which will use the field's default value if
                provided.

        Returns:
            A new dataframe with nulls filled in according to the provided `strategy`
                parameter.
        """
        if strategy != "defaults":  # pragma: no cover
            return super().fill_null(strategy=strategy)
        return self.with_columns(
            [
                pl.col(column).fill_null(pl.lit(default_value))
                for column, default_value in self.model.defaults.items()
            ]
        ).set_model(self.model)

    def get(self, predicate: Optional[pl.Expr] = None) -> ModelType:
        """
        Fetch the single row that matches the given polars predicate.

        If you expect a data frame to already consist of one single row,
        you can use get() without any arguments to return that row.

        Raises:
            RowDoesNotExist: If zero rows evaluate to true for the given predicate.
            MultipleRowsReturned: If more than one row evaluates to true for the given
                predicate.
            RuntimeError:  # noqa: DAR402
                The superclass of both `RowDoesNotExist` and `MultipleRowsReturned`
                if you want to catch both exeptions with the same class.

        Args:
            predicate: A polars expression defining the criteria of the filter.

        Returns:
            A pydantic-derived base model representing the given row.
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

    def _pydantic_model(self) -> Type[Model]:
        """
        Dynamically construct patito model compliant with dataframe.

        Returns:
            A pydantic model class where all the rows have been specified as
                `typing.Any` fields.
        """
        from patito.pydantic import Model

        pydantic_annotations = {column: (Any, ...) for column in self.columns}
        return cast(
            Type[Model],
            create_model(  # type: ignore
                "UntypedRow",
                __base__=Model,
                **pydantic_annotations,
            ),
        )

    @classmethod
    def read_csv(cls: Type[DF], *args, **kwargs) -> DF:  # noqa: ANN
        """
        Read CSV and apply correct column name and types from model.

        If any fields have `derived_from` specified, the given expression will be used
        to populate the given column.

        Args:
            *args: All positional arguments are forwarded to `polars.read_csv`.
            **kwargs: All keyword arguments are forwarded to `polars.read_csv`.

        Returns:
            A dataframe representing the given CSV file data.
        """
        kwargs.setdefault("dtypes", cls.model.dtypes)
        if not kwargs.get("has_header", True) and "columns" not in kwargs:
            kwargs.setdefault("new_columns", cls.model.columns)
        df = cls.model.DataFrame._from_pydf(pl.read_csv(*args, **kwargs)._df)
        return df.derive()
