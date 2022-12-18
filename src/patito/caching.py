"""Module containing caching utilities for polars."""
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

if TYPE_CHECKING:
    from patito import Model


P = ParamSpec("P")
DF = TypeVar("DF", bound=Union[pl.DataFrame, pl.LazyFrame], covariant=True)

# Increment this integer whenever you make backwards-incompatible changes to
# the parquet caching implemented in WrappedQueryFunc, then such caches
# are ejected the next time the wrapper tries to read from them.
CACHE_VERSION = 1


class QueryFunc(Protocol[P]):
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


class WrappedQueryFunc(Generic[P, DF]):
    """A class acting as a function that returns a polars.DataFrame when called."""

    _cache: Union[bool, Path]

    def __init__(  # noqa: C901
        self,
        wrapped_function: QueryFunc[P],
        cache_directory: Path,
        sql_to_arrow: Callable[..., pa.Table],
        ttl: timedelta,
        lazy: bool = False,
        cache: Union[str, Path, bool] = False,
        model: Union[Type["Model"], None] = None,
        query_executor_kwargs: Optional[Dict[Any, Any]] = None,
    ) -> None:
        """Convert SQL string query function to polars.DataFrame function.

        Args:
            wrapped_function: A function that takes arbitrary arguments and returns
                an SQL query string.
            cache_directory: Path to directory to store parquet cache files in.
            sql_to_arrow: Function used to execute SQL queries and return pyarrow
                Tables.
            ttl: See QueryCacher.cache for documentation.
            lazy: See QueryCacher.cache for documentation.
            cache: See QueryCacher.cache for documentation.
            model: See QueryCacher.cache for documentation.
            query_executor_kwargs: Arbitrary keyword arguments forwarded to the query
                executor, in this case sql_to_arrow, and thus db_params is the only
                keyword argument supported so far.

        Raises:
            ValueError: If the given path does not have a '.parquet' file extension.
        """
        if not isinstance(cache, bool) and Path(cache).suffix != ".parquet":
            raise ValueError("Cache paths must have the '.parquet' file extension!")

        if isinstance(cache, (Path, str)):
            self._cache = cache_directory.joinpath(cache)
        else:
            self._cache = cache
        self._wrapped_function = wrapped_function
        self.cache_directory = cache_directory

        self._query_executor_kwargs = query_executor_kwargs or {}
        # Unless explicitly specified otherwise by the end-user, we retrieve query
        # results as arrow tables with column types directly supported by polars.
        # Otherwise the resulting parquet files that are written to disk can not be
        # lazily read with polars.scan_parquet.
        self._query_executor_kwargs.setdefault("cast_to_polars_equivalent_types", True)

        # We construct the new function with the same parameter signature as
        # wrapped_function, but with polars.DataFrame as the return type.
        @wraps(wrapped_function)
        def cached_func(*args: P.args, **kwargs: P.kwargs) -> DF:
            sql_query = wrapped_function(*args, **kwargs)
            cache_path = self.cache_path(*args, **kwargs)
            if cache_path and cache_path.exists():
                metadata: Dict[bytes, bytes] = pq.read_schema(cache_path).metadata or {}

                # Check if the cache file was produced by an identical SQL query
                is_same_query = metadata.get(b"query") == sql_query.encode("utf-8")

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

            arrow_table = sql_to_arrow(sql_query, **self._query_executor_kwargs)
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
                ] = self._wrapped_function.__name__.encode("utf-8")
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
        function_signature = inspect.signature(self._wrapped_function)
        bound_arguments = function_signature.bind(*args, **kwargs)

        if isinstance(self._cache, Path):
            # Interpret relative paths relative to the main query cache directory
            return Path(str(self._cache).format(**bound_arguments.arguments))
        elif self._cache is True:
            directory: Path = self.cache_directory / self._wrapped_function.__name__
            directory.mkdir(exist_ok=True, parents=True)
            sql_query = self.sql_query(*args, **kwargs)
            sql_query_hash = hashlib.sha1(  # noqa: S324,S303
                sql_query.encode("utf-8")
            ).hexdigest()
            return directory / f"{sql_query_hash}.parquet"
        else:
            return None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> DF:  # noqa: D102
        return self._cached_func(*args, **kwargs)

    def sql_query(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """
        Return SQL query to be executed for the given parameters.

        Args:
            *args: Positional arguments used to construct the SQL query string.
            *kwargs: Keyword arguments used to construct the SQL query string.

        Returns:
            The SQL query string produced for the given input parameters.
        """
        return self._wrapped_function(*args, **kwargs)

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
                self.cache_directory / self._wrapped_function.__name__ / "*.parquet"
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
                ) == self._wrapped_function.__name__.encode("utf-8"):
                    Path(parquet_path).unlink()
            except Exception:  # noqa: S112
                # If we can't read the parquet metadata for some reason,
                # it is probably not a cache anyway.
                continue


class QueryCache:
    """
    Construct manager for executing SQL queries and caching the results.

    Args:
        sql_to_arrow: The function that the QueryCache should use for executing SQL
            queries. Its first argument should be the SQL query string to execute, and
            it should return the query result as an arrow table, for instance
            pyarrow.Table.
        cache_directory: Path to the directory where caches should be stored as parquet
            files.
        default_ttl: The default Time To Live (TTL), or with other words, how long to
            wait until caches are refreshed due to old age. The given default TTL can be
            overwritten by specifying the ``ttl`` parameter in :func:`QueryCache.query`.
            The default is 52 weeks.

    Examples:

        We start by importing the necessary modules:

        >>> from pathlib import Path
        ...
        >>> import patito as pt
        >>> import pyarrow as pa

        In order to construct a ``QueryCache``, we need to provide the constructor with
        a function that can *execute* query strings. How to construct this function will
        depend on your database of choice. For the purposes of demonstration we will use
        SQLite since it is built into Python's standard library, but you can use
        anything such as for example Snowflake or PostgresQL.

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

        >>> def query_executor(query: str) -> pa.Table:
        ...     cursor = dummy_database()
        ...     cursor.execute(query)
        ...     columns = [description[0] for description in cursor.description]
        ...     data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        ...     return pa.Table.from_pylist(data)

        We can now construct a ``QueryCache`` object, providing ``query_executor``
        as the way to execute SQL queries:

        >>> cache = pt.caching.QueryCache(
        ...     sql_to_arrow=query_executor,
        ...     cache_directory=Path("/tmp/patito_cache"),
        ... )

        The main way to use a ``QueryCache`` object is to use the ``@QueryCache.cache``
        decarator to wrap functions which return SQL query *strings*.

        >>> @cache.query()
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
        ``cache=True`` to the ``@cache.query(...)`` decorator. Other arguments are also
        accepted, such as ``lazy=True`` if you want to retrieve the results in the form
        of a ``LazyFrame`` instead of a ``DataFrame``, ``ttl`` if you want to specify
        another TTL, and any additional keyword arguments are forwarded to
        ``query_executor`` when the SQL query is executed. You can read more about these
        parameters in the documentation of :ref:`QueryCache.query`.
    """

    def __init__(
        self,
        sql_to_arrow: Callable[..., pa.Table],
        cache_directory: Path,
        default_ttl: timedelta = timedelta(weeks=52),  # noqa: B008
    ) -> None:
        self.sql_to_arrow = sql_to_arrow
        self.cache_directory = cache_directory
        self.default_ttl = default_ttl

        self.cache_directory.mkdir(exist_ok=True, parents=True)

    # With lazy = False a DataFrame-producing wrapper is returned
    @overload
    def query(
        self,
        *,
        lazy: Literal[False] = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[[QueryFunc[P]], WrappedQueryFunc[P, pl.DataFrame]]:
        ...  # pragma: no cover

    # With lazy = True a LazyFrame-producing wrapper is returned
    @overload
    def query(
        self,
        *,
        lazy: Literal[True],
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[[QueryFunc[P]], WrappedQueryFunc[P, pl.LazyFrame]]:
        ...  # pragma: no cover

    def query(
        self,
        *,
        lazy: bool = False,
        cache: Union[str, Path, bool] = False,
        ttl: Optional[timedelta] = None,
        model: Union[Type["Model"], None] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> Callable[
        [QueryFunc[P]], WrappedQueryFunc[P, Union[pl.DataFrame, pl.LazyFrame]]
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

        def wrapper(wrapped_function: QueryFunc) -> WrappedQueryFunc:
            return WrappedQueryFunc(
                wrapped_function=wrapped_function,
                lazy=lazy,
                cache=cache,
                ttl=ttl if ttl is not None else self.default_ttl,
                cache_directory=self.cache_directory,
                model=model,
                sql_to_arrow=_with_query_metadata(self.sql_to_arrow),
                query_executor_kwargs=kwargs,
            )

        return wrapper


def _with_query_metadata(sql_to_arrow: Callable[P, pa.Table]) -> Callable[P, pa.Table]:
    """
    Wrap SQL-query executor with additional logic.

    Args:
        sql_to_arrow: Function accepting an SQL query as its first argument and
            returning an Arrow table.

    Returns:
        New function that returns Arrow table with additional metedata. Arrow types
        which are not supported by polars directly have also been converted to
        compatible ones where applicable.
    """

    @wraps(sql_to_arrow)
    def wrapped_sql_to_arrow(
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> pa.Table:
        cast_to_polars_equivalent_types = kwargs.pop(
            "cast_to_polars_equivalent_types", True
        )
        start_time = datetime.now()
        arrow_table = sql_to_arrow(*args, **kwargs)
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

    return wrapped_sql_to_arrow
