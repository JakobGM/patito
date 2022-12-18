from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import polars as pl
import pytest

import patito as pt

# Python 3.7 does not support pyarrow
pa = pytest.importorskip("pyarrow")


class LoggingQueryCache(pt.caching.QueryCache):
    """A dummy query cache with an associated query execution log."""

    executed_queries: List[str]


@pytest.fixture()
def query_cache(tmp_path) -> LoggingQueryCache:
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
    def sql_to_arrow(
        query, mock_data: Optional[dict] = None
    ) -> pa.Table:  # type: ignore
        executed_queries.append(query)
        data = {"column": [1, 2, 3]} if mock_data is None else mock_data
        return pa.Table.from_pydict(data)

    query_cache = LoggingQueryCache(
        sql_to_arrow=sql_to_arrow,
        cache_directory=tmp_path,
        default_ttl=timedelta(weeks=52),
    )

    # Attach the query execution log as an attribute of the cacher
    query_cache.executed_queries = executed_queries
    return query_cache


def test_uncached_query(query_cache: LoggingQueryCache):
    """It should not cache queries by default."""

    @query_cache.query()
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


def test_cached_query(query_cache: LoggingQueryCache):
    """It should cache queries if so parametrized."""

    # We enable cache for the given query
    @query_cache.query(cache=True)
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
    query_cache: LoggingQueryCache,
    tmpdir: Path,
) -> None:
    """It should cache queries in the provided path."""
    cache_path = Path(tmpdir / "name.parquet")

    # This time we specify an explicit path
    @query_cache.query(cache=cache_path)
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

        @query_cache.query(cache=tmpdir / "name.csv")
        def products(version):
            return f"query {version}"


def test_cached_query_with_relative_path(query_cache: LoggingQueryCache) -> None:
    """Relative paths should be interpreted relative to the cache directory."""
    relative_path = Path("foo/bar.parquet")

    @query_cache.query(cache=relative_path)
    def products():
        return "query"

    products()
    assert (query_cache.cache_directory / "foo" / "bar.parquet").exists()


def test_cached_query_with_format_string(query_cache: LoggingQueryCache) -> None:
    """Strings with placeholders should be interpolated."""

    @query_cache.query(cache="version-{version}.parquet")
    def products(version: int):
        return f"query {version}"

    # It should work for both positional arguments...
    products(1)
    assert (query_cache.cache_directory / "version-1.parquet").exists()
    # ... and keywords
    products(version=2)
    assert (query_cache.cache_directory / "version-2.parquet").exists()


def test_cached_query_with_format_path(query_cache: LoggingQueryCache) -> None:
    """Paths with placeholders should be interpolated."""

    @query_cache.query(cache=query_cache.cache_directory / "version-{version}.parquet")
    def products(version: int):
        return f"query {version}"

    # It should work for both positional arguments...
    products(1)
    assert (query_cache.cache_directory / "version-1.parquet").exists()
    # ... and keywords
    products(version=2)
    assert (query_cache.cache_directory / "version-2.parquet").exists()


def test_cache_ttl(query_cache: LoggingQueryCache, monkeypatch):
    """It should automatically refresh the cache according to the TTL."""

    # We freeze the time during the execution of this test
    class FrozenDatetime:
        def __init__(self, year: int, month: int, day: int) -> None:
            self.frozen_time = datetime(year=year, month=month, day=day)
            monkeypatch.setattr(pt.caching, "datetime", self)

        def now(self):
            return self.frozen_time

        @staticmethod
        def fromisoformat(*args, **kwargs):
            return datetime.fromisoformat(*args, **kwargs)

    # The cache should be cleared every week
    @query_cache.query(cache=True, ttl=timedelta(weeks=1))
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
def test_lazy_query(query_cache: LoggingQueryCache, cache: bool):
    """It should return a LazyFrame when specified with lazy=True."""

    @query_cache.query(lazy=True, cache=cache)
    def lazy():
        return "query"

    @query_cache.query(lazy=False, cache=cache)
    def eager():
        return "query"

    # We invoke it twice, first not hitting the cache, and then hitting it
    assert lazy().collect().frame_equal(eager())
    assert lazy().collect().frame_equal(eager())


def test_model_query_model_validation(query_cache: LoggingQueryCache):
    """It should validate the data model."""

    class CorrectModel(pt.Model):
        column: int

    @query_cache.query(model=CorrectModel)
    def correct_data():
        return ""

    assert isinstance(correct_data(), pl.DataFrame)

    class IncorrectModel(pt.Model):
        column: str

    @query_cache.query(model=IncorrectModel)
    def incorrect_data():
        return ""

    with pytest.raises(pt.exceptions.ValidationError):
        incorrect_data()


def test_custom_forwarding_of_parameters_to_query_function(
    query_cache: LoggingQueryCache,
):
    """It should forward all additional parameters to the sql_to_arrow function."""

    # The dummy cacher accepts a "data" parameter, specifying the data to be returned
    data = {"actual_data": [10, 20, 30]}

    @query_cache.query(mock_data=data)
    def custom_data():
        return "select 1, 2, 3 as dummy_column"

    assert custom_data().frame_equal(pl.DataFrame(data))

    # It should also work without type normalization
    @query_cache.query(mock_data=data, cast_to_polars_equivalent_types=False)
    def non_normalized_custom_data():
        return "select 1, 2, 3 as dummy_column"

    assert non_normalized_custom_data().frame_equal(pl.DataFrame(data))


def test_clear_caches(query_cache: LoggingQueryCache):
    """It should clear all cache files with .clear_all_caches()."""

    @query_cache.query(cache=True)
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
    @query_cache.query(cache=False)
    def uncached_products(version: int):
        return f"query {version}"

    uncached_products.clear_caches()


def test_clear_caches_with_formatted_paths(query_cache: LoggingQueryCache):
    """Formatted paths should also be properly cleared."""
    # We specify another temporary cache directory to see if caches can be cleared
    # irregardless of the cache directory's location.
    tmp_dir = TemporaryDirectory()
    cache_dir = Path(tmp_dir.name)

    @query_cache.query(cache=cache_dir / "{a}" / "{b}.parquet")
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


def test_ejection_of_incompatible_caches(query_cache: LoggingQueryCache):
    """It should clear old, incompatible caches."""

    cache_path = query_cache.cache_directory / "my_cache.parquet"

    @query_cache.query(cache=cache_path)
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
        int.from_bytes(metadata[b"cache_version"], "little") == pt.caching.CACHE_VERSION
    )
    metadata[b"cache_version"] = (pt.caching.CACHE_VERSION - 1).to_bytes(
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
