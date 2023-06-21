import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional

import polars as pl
import pytest

import patito as pt

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore
else:
    # Python 3.7 does not support pyarrow
    pa = pytest.importorskip("pyarrow")


class LoggingQuerySource(pt.Database):
    """A dummy query source with an associated query execution log."""

    executed_queries: List[str]


@pytest.fixture()
def query_cache(tmp_path) -> LoggingQuerySource:
    """
    Return dummy query cache with query execution logger.

    Args:
        tmp_path: Test-specific temporary directory provided by pytest.

    Returns:
        A cacher which also keeps track of the executed queries.
    """
    # Keep track of the executed queries in a mutable list
    executed_queries = []

    # Unless other is specified, some dummy data is always returned
    def query_handler(query, mock_data: Optional[dict] = None) -> pa.Table:
        executed_queries.append(query)
        data = {"column": [1, 2, 3]} if mock_data is None else mock_data
        return pa.Table.from_pydict(data)

    query_cache = LoggingQuerySource(
        query_handler=query_handler,
        cache_directory=tmp_path,
        default_ttl=timedelta(weeks=52),
    )

    # Attach the query execution log as an attribute of the query source
    query_cache.executed_queries = executed_queries
    return query_cache


@pytest.fixture
def query_source(tmpdir) -> LoggingQuerySource:
    """
    A QuerySource connected to an in-memory SQLite3 database with dummy data.

    Args:
        tmpdir: Test-specific temporary directory provided by pytest.

    Returns:
        A query source which also keeps track of the executed queries.
    """
    # Keep track of the executed queries in a mutable list
    executed_queries = []

    def dummy_database() -> sqlite3.Cursor:
        connection = sqlite3.connect(":memory:")
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE movies(title, year, score)")
        data = [
            ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
            ("Monty Python's The Meaning of Life", 1983, 7.5),
            ("Monty Python's Life of Brian", 1979, 8.0),
        ]
        cursor.executemany("INSERT INTO movies VALUES(?, ?, ?)", data)
        connection.commit()
        return cursor

    def query_handler(query: str) -> pa.Table:
        cursor = dummy_database()
        cursor.execute(query)
        executed_queries.append(query)
        columns = [description[0] for description in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return pa.Table.from_pylist(data)

    # Attach the query execution log as an attribute of the query source
    tmp_dir = Path(tmpdir)
    query_cache = LoggingQuerySource(
        query_handler=query_handler,
        cache_directory=tmp_dir,
    )
    query_cache.executed_queries = executed_queries
    return query_cache


def test_uncached_query(query_cache: LoggingQuerySource):
    """It should not cache queries by default."""

    @query_cache.as_query()
    def products():
        return "query"

    # First time it is called we should execute the query
    products()
    assert query_cache.executed_queries == ["query"]
    # And no cache file is created
    assert not any(query_cache.cache_directory.iterdir())

    # The next time the query is executed again
    products()
    assert query_cache.executed_queries == ["query", "query"]
    # And still no cache file
    assert not any(query_cache.cache_directory.iterdir())


def test_cached_query(query_cache: LoggingQuerySource):
    """It should cache queries if so parametrized."""

    # We enable cache for the given query
    @query_cache.as_query(cache=True)
    def products(version: int):
        return f"query {version}"

    # The cache is stored in the "products" sub-folder
    cache_dir = query_cache.cache_directory / "products"

    # First time the query is executed
    products(version=1)
    assert query_cache.executed_queries == ["query 1"]
    # And the result is stored in a cache file
    assert len(list(cache_dir.iterdir())) == 1

    # The next time the query is *not* executed
    products(version=1)
    assert query_cache.executed_queries == ["query 1"]
    # And the cache file persists
    assert len(list(cache_dir.iterdir())) == 1

    # But if we change the query itself, it is executed
    products(version=2)
    assert query_cache.executed_queries == ["query 1", "query 2"]
    # And it is cached in a separate file
    assert len(list(cache_dir.iterdir())) == 2

    # If we delete the cache file, the query is re-executed
    for cache_file in cache_dir.iterdir():
        cache_file.unlink()
    products(version=1)
    assert query_cache.executed_queries == ["query 1", "query 2", "query 1"]
    # And the cache file is rewritten
    assert len(list(cache_dir.iterdir())) == 1

    # We clear the cache with .clear_cache()
    products.refresh_cache(version=1)
    assert query_cache.executed_queries == ["query 1", "query 2", "query 1", "query 1"]
    # We can also clear caches that have never existed
    products.refresh_cache(version=3)
    assert query_cache.executed_queries[-1] == "query 3"


def test_cached_query_with_explicit_path(
    query_cache: LoggingQuerySource,
    tmpdir: Path,
) -> None:
    """It should cache queries in the provided path."""
    cache_path = Path(tmpdir / "name.parquet")

    # This time we specify an explicit path
    @query_cache.as_query(cache=cache_path)
    def products(version):
        return f"query {version}"

    # At first the path does not exist
    assert not cache_path.exists()

    # We then execute and cache the query
    products(version=1)
    assert cache_path.exists()
    assert query_cache.executed_queries == ["query 1"]

    # And the next time it is reused
    products(version=1)
    assert query_cache.executed_queries == ["query 1"]
    assert cache_path.exists()

    # If the query changes, it is re-executed
    products(version=2)
    assert query_cache.executed_queries == ["query 1", "query 2"]

    # If a non-parquet file is specified, it will raise
    with pytest.raises(
        ValueError,
        match=r"Cache paths must have the '\.parquet' file extension\!",
    ):

        @query_cache.as_query(cache=tmpdir / "name.csv")
        def products(version):
            return f"query {version}"


def test_cached_query_with_relative_path(query_cache: LoggingQuerySource) -> None:
    """Relative paths should be interpreted relative to the cache directory."""
    relative_path = Path("foo/bar.parquet")

    @query_cache.as_query(cache=relative_path)
    def products():
        return "query"

    products()
    assert (query_cache.cache_directory / "foo" / "bar.parquet").exists()


def test_cached_query_with_format_string(query_cache: LoggingQuerySource) -> None:
    """Strings with placeholders should be interpolated."""

    @query_cache.as_query(cache="version-{version}.parquet")
    def products(version: int):
        return f"query {version}"

    # It should work for both positional arguments...
    products(1)
    assert (query_cache.cache_directory / "version-1.parquet").exists()
    # ... and keywords
    products(version=2)
    assert (query_cache.cache_directory / "version-2.parquet").exists()


def test_cached_query_with_format_path(query_cache: LoggingQuerySource) -> None:
    """Paths with placeholders should be interpolated."""

    @query_cache.as_query(
        cache=query_cache.cache_directory / "version-{version}.parquet"
    )
    def products(version: int):
        return f"query {version}"

    # It should work for both positional arguments...
    products(1)
    assert (query_cache.cache_directory / "version-1.parquet").exists()
    # ... and keywords
    products(version=2)
    assert (query_cache.cache_directory / "version-2.parquet").exists()


def test_cache_ttl(query_cache: LoggingQuerySource, monkeypatch):
    """It should automatically refresh the cache according to the TTL."""

    # We freeze the time during the execution of this test
    class FrozenDatetime:
        def __init__(self, year: int, month: int, day: int) -> None:
            self.frozen_time = datetime(year=year, month=month, day=day)
            monkeypatch.setattr(pt.database, "datetime", self)  # pyright: ignore

        def now(self):
            return self.frozen_time

        @staticmethod
        def fromisoformat(*args, **kwargs):
            return datetime.fromisoformat(*args, **kwargs)

    # The cache should be cleared every week
    @query_cache.as_query(cache=True, ttl=timedelta(weeks=1))
    def users():
        return "query"

    # The first time the query should be executed
    FrozenDatetime(year=2000, month=1, day=1)
    users()
    assert query_cache.executed_queries == ["query"]

    # The next time it should not be executed
    users()
    assert query_cache.executed_queries == ["query"]

    # Even if we advance the time by one day,
    # the cache should still be used.
    FrozenDatetime(year=2000, month=1, day=2)
    users()
    assert query_cache.executed_queries == ["query"]

    # Then we let one week pass, and the cache should be cleared
    FrozenDatetime(year=2000, month=1, day=8)
    users()
    assert query_cache.executed_queries == ["query", "query"]

    # But then it will be reused for another week
    users()
    assert query_cache.executed_queries == ["query", "query"]


@pytest.mark.parametrize("cache", [True, False])
def test_lazy_query(query_cache: LoggingQuerySource, cache: bool):
    """It should return a LazyFrame when specified with lazy=True."""

    @query_cache.as_query(lazy=True, cache=cache)
    def lazy():
        return "query"

    @query_cache.as_query(lazy=False, cache=cache)
    def eager():
        return "query"

    # We invoke it twice, first not hitting the cache, and then hitting it
    assert lazy().collect().frame_equal(eager())
    assert lazy().collect().frame_equal(eager())


def test_model_query_model_validation(query_cache: LoggingQuerySource):
    """It should validate the data model."""

    class CorrectModel(pt.Model):
        column: int

    @query_cache.as_query(model=CorrectModel)
    def correct_data():
        return ""

    assert isinstance(correct_data(), pl.DataFrame)

    class IncorrectModel(pt.Model):
        column: str

    @query_cache.as_query(model=IncorrectModel)
    def incorrect_data():
        return ""

    with pytest.raises(pt.exceptions.ValidationError):
        incorrect_data()


def test_custom_forwarding_of_parameters_to_query_function(
    query_cache: LoggingQuerySource,
):
    """It should forward all additional parameters to the sql_to_arrow function."""

    # The dummy cacher accepts a "data" parameter, specifying the data to be returned
    data = {"actual_data": [10, 20, 30]}

    @query_cache.as_query(mock_data=data)
    def custom_data():
        return "select 1, 2, 3 as dummy_column"

    assert custom_data().frame_equal(pl.DataFrame(data))

    # It should also work without type normalization
    @query_cache.as_query(mock_data=data, cast_to_polars_equivalent_types=False)
    def non_normalized_custom_data():
        return "select 1, 2, 3 as dummy_column"

    assert non_normalized_custom_data().frame_equal(pl.DataFrame(data))


def test_clear_caches(query_cache: LoggingQuerySource):
    """It should clear all cache files with .clear_all_caches()."""

    @query_cache.as_query(cache=True)
    def products(version: int):
        return f"query {version}"

    # The cache is stored in the "products" sub-directory
    products_cache_dir = query_cache.cache_directory / "products"

    # We produce two cache files
    products(version=1)
    products(version=2)
    assert query_cache.executed_queries == ["query 1", "query 2"]
    assert len(list(products_cache_dir.iterdir())) == 2

    # We also insert another parquet file that should *not* be deleted
    dummy_parquet_path = products_cache_dir / "dummy.parquet"
    pl.DataFrame().write_parquet(dummy_parquet_path)

    # And an invalid parquet file
    invalid_parquet_path = products_cache_dir / "invalid.parquet"
    invalid_parquet_path.write_bytes(b"invalid content")

    # We delete all caches, but not the dummy parquet file
    products.clear_caches()
    assert len(list(products_cache_dir.iterdir())) == 2
    assert dummy_parquet_path.exists()
    assert invalid_parquet_path.exists()

    # The next time both queries need to be re-executed
    products(version=1)
    products(version=2)
    assert query_cache.executed_queries == ["query 1", "query 2"] * 2
    assert len(list(products_cache_dir.iterdir())) == 4

    # If caching is not enabled, clear_caches should be a NO-OP
    @query_cache.as_query(cache=False)
    def uncached_products(version: int):
        return f"query {version}"

    uncached_products.clear_caches()


def test_clear_caches_with_formatted_paths(query_cache: LoggingQuerySource):
    """Formatted paths should also be properly cleared."""
    # We specify another temporary cache directory to see if caches can be cleared
    # irregardless of the cache directory's location.
    tmp_dir = TemporaryDirectory()
    cache_dir = Path(tmp_dir.name)

    @query_cache.as_query(cache=cache_dir / "{a}" / "{b}.parquet")
    def users(a: int, b: int):
        return f"query {a}.{b}"

    users(1, 1)
    users(1, 2)
    users(2, 1)

    assert query_cache.executed_queries == ["query 1.1", "query 1.2", "query 2.1"]

    assert {str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*")} == {
        # Both directories have been created
        "1",
        "2",
        # Two cache files for a=1
        "1/1.parquet",
        "1/2.parquet",
        # One cache file for a=2
        "2/1.parquet",
    }

    # We insert another parquet file that should *not* be cleared
    pl.DataFrame().write_parquet(cache_dir / "1" / "3.parquet")

    # Only directories and non-cached files should be kept
    users.clear_caches()
    assert {str(path.relative_to(cache_dir)) for path in cache_dir.rglob("*")} == {
        "1",
        "2",
        "1/3.parquet",
    }
    tmp_dir.cleanup()


def test_ejection_of_incompatible_caches(query_cache: LoggingQuerySource):
    """It should clear old, incompatible caches."""

    cache_path = query_cache.cache_directory / "my_cache.parquet"

    @query_cache.as_query(cache=cache_path)
    def my_query():
        return "my query"

    # Write a parquet file without any metadata
    pl.DataFrame().write_parquet(cache_path)

    # The existing parquet file without metadata should be overwritten
    df = my_query()
    assert not df.is_empty()
    assert query_cache.executed_queries == ["my query"]

    # Now we decrement the version number of the cache in order to overwrite it
    arrow_table = pa.parquet.read_table(cache_path)  # noqa
    metadata = arrow_table.schema.metadata
    assert (
        int.from_bytes(metadata[b"cache_version"], "little")
        == pt.database.CACHE_VERSION  # pyright: ignore
    )
    metadata[b"cache_version"] = (
        pt.database.CACHE_VERSION - 1  # pyright: ignore
    ).to_bytes(
        length=16,
        byteorder="little",
        signed=False,
    )
    pa.parquet.write_table(
        arrow_table.replace_schema_metadata(metadata),
        where=cache_path,
    )

    # The query should now be re-executed
    my_query()
    assert query_cache.executed_queries == ["my query"] * 2

    # Deleting the cache_version alltogether should also retrigger the query
    del metadata[b"cache_version"]
    pa.parquet.write_table(
        arrow_table.replace_schema_metadata(metadata),
        where=cache_path,
    )
    my_query()
    assert query_cache.executed_queries == ["my query"] * 3


def test_adherence_to_xdg_directory_standard(monkeypatch, tmpdir):
    """It should use XDG Cache Home when no cache directory is specified."""
    xdg_cache_home = tmpdir / ".cache"
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)
    query_source = pt.Database(query_handler=lambda query: pa.Table())
    assert query_source.cache_directory == xdg_cache_home / "patito"

    del os.environ["XDG_CACHE_HOME"]
    query_source = pt.Database(query_handler=lambda query: pa.Table())
    assert query_source.cache_directory == Path("~/.cache/patito").resolve()


def test_invoking_query_source_directly_with_query_string(
    query_source: LoggingQuerySource,
):
    """It should accept SQL queries directly, not ony query constructors."""
    sql = "select * from movies"
    movies = query_source.query(sql)
    assert query_source.executed_queries == [sql]
    assert len(list(query_source.cache_directory.iterdir())) == 0
    assert movies.height == 3

    for _ in range(2):
        query_source.query(sql, cache=True)
        assert query_source.executed_queries == [sql] * 2
        assert (
            len(list((query_source.cache_directory / "__direct_query").iterdir())) == 1
        )

    assert query_source.query(sql, lazy=True).collect().frame_equal(movies)


@pytest.mark.skip(reason="TODO: Future feature to implement")
def test_custom_kwarg_hashing(tmp_path):
    """You should be able to hash the keyword arguments passed to the query handler."""

    executed_queries = []

    def query_handler(query: str, prod=False) -> pa.Table:
        executed_queries.append(query)
        return pa.Table.from_pydict({"column": [1, 2, 3]})

    def query_handler_hasher(query: str, prod: bool) -> bytes:
        return bytes(prod)

    dummy_source = pt.Database(
        query_handler=query_handler,
        query_handler_hasher=query_handler_hasher,  # pyright: ignore
        cache_directory=tmp_path,
    )

    # The first time the query should be executed
    sql_query = "select * from my_table"
    dummy_source.query(sql_query, cache=True)
    assert executed_queries == [sql_query]
    assert len(list(dummy_source.cache_directory.rglob("*.parquet"))) == 1

    # The second time the dev query has been cached
    dummy_source.query(sql_query, cache=True)
    assert executed_queries == [sql_query]
    assert len(list(dummy_source.cache_directory.rglob("*.parquet"))) == 1

    # The production query has never executed, so a new query is executed
    dummy_source.query(sql_query, cache=True, prod=True)
    assert executed_queries == [sql_query] * 2
    assert len(list(dummy_source.cache_directory.rglob("*.parquet"))) == 2

    # Then the production query cache is used
    dummy_source.query(sql_query, cache=True, prod=True)
    assert executed_queries == [sql_query] * 2
    assert len(list(dummy_source.cache_directory.rglob("*.parquet"))) == 2

    # And the dev query cache still remains
    dummy_source.query(sql_query, cache=True)
    assert executed_queries == [sql_query] * 2
    assert len(list(dummy_source.cache_directory.rglob("*.parquet"))) == 2
