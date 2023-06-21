"""Module containing utilities for retrieving data from external databases."""
import glob
import hashlib
import inspect
import re
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import polars as pl
import pyarrow as pa  # type: ignore[import]
import pyarrow.parquet as pq  # type: ignore[import]
from typing_extensions import Literal, ParamSpec, Protocol

from patito import xdg

if TYPE_CHECKING:
    from patito import Model


P = ParamSpec("P")
DF = TypeVar("DF", bound=Union[pl.DataFrame, pl.LazyFrame], covariant=True)

# Increment this integer whenever you make backwards-incompatible changes to
# the parquet caching implemented in WrappedQueryFunc, then such caches
# are ejected the next time the wrapper tries to read from them.
CACHE_VERSION = 1


class QueryConstructor(Protocol[P]):
    """A function taking arbitrary arguments and returning an SQL query string."""

    __name__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """
        Return SQL query constructed from the given parameters.

        Args:
            *args: Positional arguments used to build SQL query.
            **kwargs: Keyword arguments used to build SQL query.
        """
        ...  # pragma: no cover


class DatabaseQuery(Generic[P, DF]):
    """A class acting as a function that returns a polars.DataFrame when called."""

    _cache: Union[bool, Path]

    def __init__(  # noqa: C901
        self,
        query_constructor: QueryConstructor[P],
        cache_directory: Path,
        query_handler: Callable[..., pa.Table],
        ttl: timedelta,
        lazy: bool = False,
        cache: Union[str, Path, bool] = False,
        model: Union[Type["Model"], None] = None,
        query_handler_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """
        Convert SQL string query function to polars.DataFrame function.

        Args:
            query_constructor: A function that takes arbitrary arguments and returns
                an SQL query string.
            cache_directory: Path to directory to store parquet cache files in.
            query_handler: Function used to execute SQL queries and return arrow
                tables.
            ttl: See Database.query for documentation.
            lazy: See Database.query for documentation.
            cache: See Database.query for documentation.
            model: See Database.query for documentation.
            query_handler_kwargs: Arbitrary keyword arguments forwarded to the provided
                query handler.

        Raises:
            ValueError: If the given path does not have a '.parquet' file extension.
        """
        if not isinstance(cache, bool) and Path(cache).suffix != ".parquet":
            raise ValueError("Cache paths must have the '.parquet' file extension!")

        if isinstance(cache, (Path, str)):
            self._cache = cache_directory.joinpath(cache)
        else:
            self._cache = cache
        self._query_constructor = query_constructor
        self.cache_directory = cache_directory

        self._query_handler_kwargs = query_handler_kwargs or {}
        # Unless explicitly specified otherwise by the end-user, we retrieve query
        # results as arrow tables with column types directly supported by polars.
        # Otherwise the resulting parquet files that are written to disk can not be
        # lazily read with polars.scan_parquet.
        self._query_handler_kwargs.setdefault("cast_to_polars_equivalent_types", True)

        # We construct the new function with the same parameter signature as
        # wrapped_function, but with polars.DataFrame as the return type.
        @wraps(query_constructor)
        def cached_func(*args: P.args, **kwargs: P.kwargs) -> DF:
            query = query_constructor(*args, **kwargs)
            cache_path = self.cache_path(*args, **kwargs)
            if cache_path and cache_path.exists():
                metadata: Dict[bytes, bytes] = pq.read_schema(cache_path).metadata or {}

                # Check if the cache file was produced by an identical SQL query
                is_same_query = metadata.get(b"query") == query.encode("utf-8")

                # Check if the cache is too old to be re-used
                cache_created_time = datetime.fromisoformat(
                    metadata.get(
                        b"query_start_time", b"1900-01-01T00:00:00.000000"
                    ).decode("utf-8")
                )
                is_fresh_cache = (datetime.now() - cache_created_time) < ttl

                # Check if the cache was produced by an incompatible version
                cache_version = int.from_bytes(
                    metadata.get(
                        b"cache_version",
                        (0).to_bytes(length=16, byteorder="little", signed=False),
                    ),
                    byteorder="little",
                    signed=False,
                )
                is_compatible_version = cache_version >= CACHE_VERSION

                if is_same_query and is_fresh_cache and is_compatible_version:
                    if lazy:
                        return pl.scan_parquet(cache_path)  # type: ignore
                    else:
                        return pl.read_parquet(cache_path)  # type: ignore

            arrow_table = query_handler(query, **self._query_handler_kwargs)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # We write the cache *before* any potential model validation since
                # we don't want to lose the result of an expensive query just because
                # the model specification is wrong.
                # We also use pyarrow.parquet.write_table instead of
                # polars.write_parquet since we want to write the arrow table's metadata
                # to the parquet file, such as the executed query, time, etc..
                # This metadata is not preserved by polars.
                metadata = arrow_table.schema.metadata
                metadata[
                    b"wrapped_function_name"
                ] = self._query_constructor.__name__.encode("utf-8")
                # Store the cache version as an 16-bit unsigned little-endian number
                metadata[b"cache_version"] = CACHE_VERSION.to_bytes(
                    length=16,
                    byteorder="little",
                    signed=False,
                )
                pq.write_table(
                    table=arrow_table.replace_schema_metadata(metadata),
                    where=cache_path,
                    # In order to support nanosecond-resolution timestamps, we must
                    # use parquet version >= 2.6.
                    version="2.6",
                )

            polars_df = cast(pl.DataFrame, pl.from_arrow(arrow_table))
            if model:
                model.validate(polars_df)

            if lazy:
                if cache_path:
                    # Delete in-memory representation of data and read from the new
                    # parquet file instead. That way we get consistent memory pressure
                    # the first and subsequent times this function is invoked.
                    del polars_df, arrow_table
                    return pl.scan_parquet(file=cache_path)  # type: ignore
                else:
                    return polars_df.lazy()  # type: ignore
            else:
                return polars_df  # type: ignore

        self._cached_func = cached_func

    def cache_path(self, *args: P.args, **kwargs: P.kwargs) -> Optional[Path]:
        """
        Return the path to the parquet cache that would store the result of the query.

        Args:
            args: The positional arguments passed to the wrapped function.
            kwargs: The keyword arguments passed to the wrapped function.

        Returns:
            A deterministic path to a parquet cache. None if caching is disabled.
        """
        # We convert args+kwargs to kwargs-only and use it to format the string
        function_signature = inspect.signature(self._query_constructor)
        bound_arguments = function_signature.bind(*args, **kwargs)

        if isinstance(self._cache, Path):
            # Interpret relative paths relative to the main query cache directory
            return Path(str(self._cache).format(**bound_arguments.arguments))
        elif self._cache is True:
            directory: Path = self.cache_directory / self._query_constructor.__name__
            directory.mkdir(exist_ok=True, parents=True)
            sql_query = self.query_string(*args, **kwargs)
            sql_query_hash = hashlib.sha1(  # noqa: S324,S303
                sql_query.encode("utf-8")
            ).hexdigest()
            return directory / f"{sql_query_hash}.parquet"
        else:
            return None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> DF:  # noqa: D102
        return self._cached_func(*args, **kwargs)

    def query_string(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """
        Return the query to be executed for the given parameters.

        Args:
            *args: Positional arguments used to construct the query string.
            *kwargs: Keyword arguments used to construct the query string.

        Returns:
            The query string produced for the given input parameters.
        """
        return self._query_constructor(*args, **kwargs)

    def refresh_cache(self, *args: P.args, **kwargs: P.kwargs) -> DF:
        """
        Force query execution by refreshing the cache.

        Args:
            *args: Positional arguments used to construct the SQL query string.
            *kwargs: Keyword arguments used to construct the SQL query string.

        Returns:
            A DataFrame representing the result of the newly executed query.
        """
        cache_path = self.cache_path(*args, **kwargs)
        if cache_path and cache_path.exists():
            cache_path.unlink()
        return self._cached_func(*args, **kwargs)

    def clear_caches(self) -> None:
        """Delete all parquet cache files produced by this query wrapper."""
        if self._cache is False:
            # Caching is not enabled, so this is simply a no-op
            return

        if self._cache is True:
            glob_pattern = str(
                self.cache_directory / self._query_constructor.__name__ / "*.parquet"
            )
        else:
            # We replace all formatting specifiers of the form '{variable}' with
            # recursive globs '**' (in case strings containing '/' are inserted) and
            # search for all occurrences of such file paths.
            # For example if cache="{a}/{b}.parquet" is specified, we search for
            # all files matching the glob pattern '**/**.parquet'.
            glob_pattern = re.sub(  # noqa: PD005
                # We specify the reluctant qualifier (?) in order to get narrow matches
                pattern=r"\{.+?\}",
                repl="**",
                string=str(self._cache),
            )

        for parquet_path in glob.iglob(glob_pattern):
            try:
                metadata: Dict[bytes, bytes] = (
                    pq.read_schema(where=parquet_path).metadata or {}
                )
                if metadata.get(
                    b"wrapped_function_name"
                ) == self._query_constructor.__name__.encode("utf-8"):
                    Path(parquet_path).unlink()
            except Exception:  # noqa: S112
                # If we can't read the parquet metadata for some reason,
                # it is probably not a cache anyway.
                continue


class Database:
    """
    Construct manager for executing SQL queries and caching the results.

    Args:
        query_handler: The function that the Database object should use for executing
            SQL queries. Its first argument should be the SQL query string to execute,
            and it should return the query result as an arrow table, for instance
            pyarrow.Table.
        cache_directory: Path to the directory where caches should be stored as parquet
            files. If not provided, the `XDG`_ Base Directory Specification will be
            used to determine the suitable cache directory, by default
            ``~/.cache/patito`` or ``${XDG_CACHE_HOME}/patito``.
        default_ttl: The default Time To Live (TTL), or with other words, how long to
            wait until caches are refreshed due to old age. The given default TTL can be
            overwritten by specifying the ``ttl`` parameter in
            :func:`Database.query`. The default is 52 weeks.

    Examples:
        We start by importing the necessary modules:

        >>> from pathlib import Path
        ...
        >>> import patito as pt
        >>> import pyarrow as pa

        In order to construct a ``Database``, we need to provide the constructor with
        a function that can *execute* query strings. How to construct this function will
        depend on what you actually want to run your queries against, for example a
        local or remote database. For the purposes of demonstration we will use
        SQLite since it is built into Python's standard library, but you can use
        anything; for example Snowflake or PostgresQL.

        We will use Python's standard library
        `documentation <https://docs.python.org/3/library/sqlite3.html>`_
        to create an in-memory SQLite database.
        It will contain a single table named ``movies`` containing some dummy data.
        The details do not really matter here, the only important part is that we
        construct a database which we can run SQL queries against.

        >>> import sqlite3
        ...
        >>> def dummy_database() -> sqlite3.Cursor:
        ...     connection = sqlite3.connect(":memory:")
        ...     cursor = connection.cursor()
        ...     cursor.execute("CREATE TABLE movies(title, year, score)")
        ...     data = [
        ...         ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
        ...         ("Monty Python's The Meaning of Life", 1983, 7.5),
        ...         ("Monty Python's Life of Brian", 1979, 8.0),
        ...     ]
        ...     cursor.executemany("INSERT INTO movies VALUES(?, ?, ?)", data)
        ...     connection.commit()
        ...     return cursor

        Using this dummy database, we are now able to construct a function which accepts
        SQL queries as its first parameter, executes the query, and returns the query
        result in the form of an Arrow table.

        >>> def query_handler(query: str) -> pa.Table:
        ...     cursor = dummy_database()
        ...     cursor.execute(query)
        ...     columns = [description[0] for description in cursor.description]
        ...     data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        ...     return pa.Table.from_pylist(data)

        We can now construct a ``Database`` object, providing ``query_handler``
        as the way to execute SQL queries.

        >>> db = pt.Database(query_handler=query_handler)

        The resulting object can now be used to execute SQL queries against the database
        and return the result in the form of a polars ``DataFrame`` object.

        >>> db.query("select * from movies order by year limit 1")
        shape: (1, 3)
        ┌──────────────────────────────┬──────┬───────┐
        │ title                        ┆ year ┆ score │
        │ ---                          ┆ ---  ┆ ---   │
        │ str                          ┆ i64  ┆ f64   │
        ╞══════════════════════════════╪══════╪═══════╡
        │ Monty Python's Life of Brian ┆ 1979 ┆ 8.0   │
        └──────────────────────────────┴──────┴───────┘

        But the main way to use a ``Database`` object is to use the
        ``@Database.as_query`` decarator to wrap functions which return SQL
        query *strings*.

        >>> @db.as_query()
        >>> def movies(newer_than_year: int):
        ...     return f"select * from movies where year > {newer_than_year}"

        This decorator will convert the function from producing query strings, to
        actually executing the given query and return the query result in the form of
        a polars ``DataFrame`` object.

        >>> movies(newer_than_year=1980)
        shape: (2, 3)
        ┌─────────────────────────────────────┬──────┬───────┐
        │ title                               ┆ year ┆ score │
        │ ---                                 ┆ ---  ┆ ---   │
        │ str                                 ┆ i64  ┆ f64   │
        ╞═════════════════════════════════════╪══════╪═══════╡
        │ Monty Python Live at the Hollywo... ┆ 1982 ┆ 7.9   │
        ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌╌┤
        │ Monty Python's The Meaning of Li... ┆ 1983 ┆ 7.5   │
        └─────────────────────────────────────┴──────┴───────┘

        Caching is not enabled by default, but it can be enabled by specifying
        ``cache=True`` to the ``@db.as_query(...)`` decorator. Other arguments are
        also accepted, such as ``lazy=True`` if you want to retrieve the results in the
        form of a ``LazyFrame`` instead of a ``DataFrame``, ``ttl`` if you want to
        specify another TTL, and any additional keyword arguments are forwarded to
        ``query_executor`` when the SQL query is executed. You can read more about these
        parameters in the documentation of :func:`Database.query`.

    .. _XDG: https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
    """

    Query = DatabaseQuery

    def __init__(  # noqa: D107
        self,
        query_handler: Callable[..., pa.Table],
        cache_directory: Optional[Path] = None,
        default_ttl: timedelta = timedelta(weeks=52),  # noqa: B008
    ) -> None:
        self.query_handler = query_handler
        self.cache_directory = cache_directory or xdg.cache_home(application="patito")
        self.default_ttl = default_ttl

        self.cache_directory.mkdir(exist_ok=True, parents=True)

    # With lazy = False a DataFrame-producing wrapper is returned
    @overload
    def as_query(
        self,
        *,
        lazy: Literal[False] = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[[QueryConstructor[P]], DatabaseQuery[P, pl.DataFrame]]:
        ...  # pragma: no cover

    # With lazy = True a LazyFrame-producing wrapper is returned
    @overload
    def as_query(
        self,
        *,
        lazy: Literal[True],
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[[QueryConstructor[P]], DatabaseQuery[P, pl.LazyFrame]]:
        ...  # pragma: no cover

    def as_query(
        self,
        *,
        lazy: bool = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[
        [QueryConstructor[P]], DatabaseQuery[P, Union[pl.DataFrame, pl.LazyFrame]]
    ]:
        """
        Execute the returned query string and return a polars dataframe.

        Args:
            lazy: If the result should be returned as a LazyFrame rather than a
                DataFrame. Allows more efficient reading from parquet caches if caching
                is enabled.
            cache: If queries should be cached in order to save time and costs.
                The cache will only be used if the exact same SQL string has
                been executed before.
                If the parameter is specified as ``True``, a parquet file is
                created for each unique query string, and is located at:
                artifacts/query_cache/<function_name>/<query_md5_hash>.parquet
                If the a string or ``pathlib.Path`` object is provided, the given path
                will be used, but it must have a '.parquet' file extension.
                Relative paths are interpreted relative to artifacts/query_cache/
                in the workspace root. The given parquet path will be overwritten
                if the query string changes, so only the latest query string value
                will be cached.
            ttl: The Time to Live (TTL) of the cache specified as a datetime.timedelta
                object. When the cache becomes older than the specified TTL, the query
                will be re-executed on the next invocation of the query function
                and the cache will refreshed.
            model: An optional Patito model used to validate the content of the
                dataframe before return.
            **kwargs: Connection parameters forwarded to sql_to_polars, for example
                db_params.

        Returns:
            A new function which returns a polars DataFrame based on the query
            specified by the original function's return string.
        """

        def wrapper(query_constructor: QueryConstructor) -> DatabaseQuery:
            return self.Query(
                query_constructor=query_constructor,
                lazy=lazy,
                cache=cache,
                ttl=ttl if ttl is not None else self.default_ttl,
                cache_directory=self.cache_directory,
                model=model,
                query_handler=_with_query_metadata(self.query_handler),
                query_handler_kwargs=kwargs,
            )

        return wrapper

    # With lazy=False, a DataFrame is returned
    @overload
    def query(
        self,
        query: str,
        *,
        lazy: Literal[False] = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> pl.DataFrame:
        ...  # pragma: no cover

    # With lazy=True, a LazyFrame is returned
    @overload
    def query(
        self,
        query: str,
        *,
        lazy: Literal[True],
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> pl.LazyFrame:
        ...  # pragma: no cover

    def query(
        self,
        query: str,
        *,
        lazy: bool = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Execute the given query and return the query result as a DataFrame or LazyFrame.

        See :ref:`Database.as_query` for a more powerful way to build and execute
        queries.

        Args:
            query: The query string to be executed, for instance an SQL query.
            lazy: If the query  result should be returned in the form of a LazyFrame
                instead of a DataFrame.
            cache: If the query result should be saved and re-used the next time the
                same query is executed. Can also be provided as a path. See
                :func:`Database.as_query` for full documentation.
            ttl: How long to use cached results until the query is re-executed anyway.
            model: A :ref:`Model` to optionally validate the query result.
            **kwargs: All additional keyword arguments are forwarded to the query
                handler which executes the given query.

        Returns:
            The result of the query in the form of a ``DataFrame`` if ``lazy=False``, or
            a ``LazyFrame`` otherwise.

        Examples:
            We will use DuckDB as our example database.

            >>> import duckdb
            >>> import patito as pt

            We will construct a really simple query source from an in-memory database.

            >>> db = duckdb.connect(":memory:")
            >>> query_handler = lambda query: db.cursor().query(query).arrow()
            >>> query_source = pt.Database(query_handler=query_handler)

            We can now use :func:`Database.query` in order to execute queries against
            the in-memory database.

            >>> query_source.query("select 1 as a, 2 as b, 3 as c")
            shape: (1, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ i32 ┆ i32 ┆ i32 │
            ╞═════╪═════╪═════╡
            │ 1   ┆ 2   ┆ 3   │
            └─────┴─────┴─────┘
        """

        def __direct_query() -> str:
            """
            A regular named function in order to store parquet files correctly.

            Returns:
                The user-provided query string.
            """
            return query

        return self.as_query(
            lazy=lazy,  # type: ignore
            cache=cache,
            ttl=ttl,
            model=model,
            **kwargs,
        )(__direct_query)()


def _with_query_metadata(query_handler: Callable[P, pa.Table]) -> Callable[P, pa.Table]:
    """
    Wrap SQL-query handler with additional logic.

    Args:
        query_handler: Function accepting an SQL query as its first argument and
            returning an Arrow table.

    Returns:
        New function that returns Arrow table with additional metedata. Arrow types
        which are not supported by polars directly have also been converted to
        compatible ones where applicable.
    """

    @wraps(query_handler)
    def wrapped_query_handler(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> pa.Table:
        cast_to_polars_equivalent_types = kwargs.pop(
            "cast_to_polars_equivalent_types", True
        )
        start_time = datetime.now()
        arrow_table = query_handler(*args, **kwargs)
        finish_time = datetime.now()
        metadata: dict = arrow_table.schema.metadata or {}
        if cast_to_polars_equivalent_types:
            # We perform a round-trip to polars and back in order to get an arrow table
            # with column types that are directly supported by polars.
            arrow_table = pl.from_arrow(arrow_table).to_arrow()

        # Store additional metadata which is useful when the arrow table is written to a
        # parquet file as a caching mechanism.
        metadata.update(
            {
                "query": args[0],
                "query_start_time": start_time.isoformat(),
                "query_end_time": finish_time.isoformat(),
            }
        )
        return arrow_table.replace_schema_metadata(metadata)

    return wrapped_query_handler


__all__ = ["Database"]
