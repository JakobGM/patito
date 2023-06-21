Integrating Polars With an SQL Database Using Efficient Caching
===============================================================

.. mermaid::
   :align: center

    %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#FFF5E6', 'secondaryColor': '#FFF5E6' }}}%%
    graph LR;
        source[Source system]
        dw[Data Warehouse]
        csv[Local .csv]
        ds[Data application]
        source-->|data pipeline|dw-->|SQL query|csv-->|read_csv|ds
        dw-->|cached Patito<br>integration|ds

Many data-driven applications involve data that must be retrieved from a database, either a remote data warehouse solution such as Databricks or Snowflake, or a local database such as DuckDB or SQLite3.
Patito offers a database-agnostic API to query such sources, returning the result as a polars DataFrame, while offering intelligent query caching on top.
By the end of this tutorial you will be able to write data ingestion logic that looks like this:

.. code::

   from typing import Optional

   from db import my_database

   @my_database.as_query(cache=True)
   def users(country: Optional[str] = None):
      query = "select * from users"
      if country:
         query += f" where country = '{country}'"
      return query

The wrapped ``users`` function will now construct, execute, cache, and return the results of the SQL query in the form of a ``polars.DataFrame`` object.
The cache for ``users(country="NO")`` will be stored independently from ``users(country="US")``, and so on.
This, among with other functionality that will be explained later, allows you to integrate your local data pipeline with your remote database in an effortless way.

The following tutorial will explain how to construct a :class:`patito.Database` object which provides Patito with the required context to execute SQL queries against your database of choice.
In turn :func:`patito.Database.query` can be used to execute SQL query strings directly and :func:`patito.Database.as_query` can be used to wrap functions that *produce* SQL query strings.
The latter decorator turns functions into :class:`patito.Database.Query <patito.database.DatabaseQuery>` objects which act very much like the original functions, only that they actually execute the constructed queries and return the results as DataFrames when invoked.
The ``Query`` object also has :ref:`additional methods <QueryMethods>` for managing the query caches and more.

This tutorial will take a relatively opinionated approach to how to organize your code.
For less opinionated documentation, see the referenced classes and methods above.

.. contents:: Table of Contents
   :local:

Setup
-----

The following tutorial will depend on ``patito`` having been installed with the ``caching`` extension group:

.. code::

   pip install patito[caching]

Code samples in this tutorial will use `DuckDB <https://duckdb.org/>`_, but you should be able to replace it with your database of choice as you follow along:

.. code::

   pip install duckdb


Construct a ``patito.Database`` Object
--------------------------------------

To begin we need to provide Patito with the tools required to query your database of choice.
First we must implement a *query handler*, a function that takes a query string as its first argument, executes the query, and returns the result of the query in the form an Arrow table.

We are going to use DuckDB as our detailed example in this tutorial, but example code for other databases, including SQLite3, is provided at the end of this section.
We start by creating a ``db.py`` module in the root of our application, and implement ``db.connection`` as a way to connect to a DuckDB instance.

.. code-block::
   :caption: **db.py** -- connection

   import duckdb

   def connection(name: str) -> duckdb.DuckDBPyConnection:
      return duckdb.connect(name)

Here ``db.connection()`` must be provided with a name, either ``:memory:`` to store the data in-memory, or a file name to persist the data on-disk.
We can use this new function in order to implement our query handler.

.. code-block::
   :caption: **db.py** - query_handler
   :emphasize-lines: 2,7-9

   import duckdb
   import pyarrow as pa

   def connection(name: str) -> duckdb.DuckDBPyConnection:
      return duckdb.connect(name)

   def query_handler(query: str, *, name: str = ":memory:") -> pa.Table:
       connection = connection(name=name)
       return connection.cursor().query(query).arrow()

Notice how the first argument of ``query_handler`` is the query string to be executed, as required by Patito, but the ``name`` keyword is specific to our database of choice.
It is now simple for us to create a :class:`patito.Database` object by providing ``db.query_handler``:

.. code-block::
   :caption: **db.py** -- pt.Database
   :emphasize-lines: 4,14

   from pathlib import Path

   import duckdb
   import patito as pt
   import pyarrow as pa

   def connection(name: str) -> duckdb.DuckDBPyConnection:
      return duckdb.connect(name)

   def query_handler(query: str, name: str = ":memory:") -> pa.Table:
       cursor = connection(name).cursor()
       return cursor.query(query).arrow()

   my_database = pt.Database(query_handler=query_handler)

Additional arguments can be provided to the ``Database`` constructor, for example a custom cache directory.
These additional parameters are documented :ref:`here <Database.__init__>`.
Documentation for constructing query handlers and :class:`patito.Database` objects for other databases is provided in the collapsable sections below:

.. collapse:: SQLite3

   See "Examples" section of :class:`patito.Database`.

.. collapse:: Other

   You are welcome to create `a GitHub issue <https://github.com/kolonialno/patito/issues/new>`_ if you need help integrating with you specific database of choice.

|
Querying the Database Directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``db`` module is now complete and we should be able to use it in order to execute queries directly against our in-memory database.

.. code-block::

   >>> from db import my_database
   >>> my_database.query("select 1 as a, 2 as b")
   shape: (1, 2)
   ┌─────┬─────┐
   │ a   ┆ b   │
   │ --- ┆ --- │
   │ i32 ┆ i32 │
   ╞═════╪═════╡
   │ 1   ┆ 2   │
   └─────┴─────┘

The query result is provided in the form of a polars ``DataFrame`` object.
Additional parameters can be provided to :func:`patito.Database.query` as described :ref:`here <Database.query>`.
As an example, the query result can be provided as a ``polars.LazyFrame`` by specifying ``lazy=True``.

.. code-block::

   >>> from db import my_database
   >>> my_database.query("select 1 as a, 2 as b", lazy=True)
   <polars.LazyFrame object at 0x13571D310>

Any *additional* keyword arguments provided to :func:`patito.Database.query` are forwarded directly to the original query handler, so the following will execute the query against the database stored in ``my.db``:

.. code-block::

   >>> my_database.query("select * from my_table", name="my.db")


.. mermaid::
   :caption: Delegation of parameters provided to :func:`patito.Database.query`.
   :align: center

    %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#FFF5E6', 'secondaryColor': '#FFF5E6' }}}%%
    graph LR;
        input["<code>Database.query(query, lazy, cache, ttl, model, **kwargs)</code>"]
        query[patito.Query]
        query_handler[Database query handler]
        input-->|<code>lazy, cache, ttl, model</code>|query
        input-->|<code>query, **kwargs</code>|query_handler

Wrapping Query-Producing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's assume that you have a project named ``user-analyzer`` which analyzes users.
The associated python package should therefore be named ``user_analyzer``.
By convention, functions for retrieving data from a remote database should be placed in the ``user_analyzer.fetch`` sub-module.
Using this module should be as simple as...

.. code::

   from user_analyzer import fetch

   user_df = fetch.users()

Start by creating the python file for the ``fetch`` sub-module, it should be located at ``projects/user-analyzer/user_analyzer/fetch.py``.
Next, implement the ``users`` function as a function that returns a SQL query that should produce the intended data when executed in the remote database...

.. code::

   def users():
       return "select * from d_users"

This is clearly not enough, the ``fetch.users`` function only returns a query string for now, but it can trivially be converted to a function that returns a dataframe instead by using the ``query`` decorator from ``db``...

.. code::

   from db import query

   @query()
   def users():
       return "select * from d_users"


Polars vs. Pandas
~~~~~~~~~~~~~~~~~

When ``user_analyzer.fetch.users()`` is invoked it will return a polars DataFrame by default.
`Polars <https://github.com/pola-rs/polars>`_ is a DataFrame library that is highly recommended over pandas in Oda; it will be familiar to most pandas users and can be easily converted to pandas when needed.
You can find introductory documentation for polars `here <https://pola-rs.github.io/polars-book/user-guide/>`_.
If you still prefer to use pandas you can use the ``.to_pandas()`` method like this...

.. code::

   from user_analyzer import fetch

   # This returns a polars DataFrame
   user_df = fetch.users()

   # This returns a pandas DataFrame
   user_df = fetch.users().to_pandas()

We can also add parameters to the ``users`` function, if needed, let's say we want to be able to filter on the users' country codes:

.. code::

   from typing import Literal, Optional

   from db import query

   @query()
   def users(country: Optional[str] = None):
       if country_code:
           return f"select * from d_users where country_code = '{country}'"
       else:
           return "select * from d_users"

You can now construct a DataFrame of all Finish users by writing ``fetch.users(country="FI")``.
If you want to access the SQL query rather than executing it, you can retrieve it with ``fetch.users.query_string(country="FI")``.

Specifying custom database parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``@query`` decorator will by default execute your SQL query against the ``ANALYTICS.ANALYTICS`` database schema.
If your query needs to use different schema, warehouses, users, etc., you can specify a custom ``db_params`` parameter to the query decorator.

Here is an example where we execute the query against ``ANALYTICTS.ANALYTICS_FORECASTING`` instead of ``ANALYTICS.ANALYTICS``.

.. code::

   from db import query

   FORECASTING_SCHEMA = {"schema": "ANALYTICS_FORECASTING"}


   @query(db_params=FORECASTING_SCHEMA):
   def covid_cases():
       return "return * from stg_covid_cases"

Normalizing column types
~~~~~~~~~~~~~~~~~~~~~~~~

A Snowflake query might produce different column types based on how many rows are returned and/or the value bounds of each column.
In order to ensure consistent behavior, ``db.query`` by default _upcasts_ all lower-typed dtypes such as ``Int8`` to ``Int64``, ``Float16`` to ``Float64``, and so on.
This behavior can be disabled by providing ``normalize_column_types=False`` to the ``@query`` decorator.

.. code::

   from db import query

   @query(normalize_column_types=False)
   def example_query():
       return "example query"

Cache Your Queries to Speed Up Data Retrieval
---------------------------------------------

Some database queries may take a long time to execute due to the data set being large and/or the computations being intensive.
In those cases you might want to store the result for reuse rather than re-executing the query every single time you invoke ``fetch.X()``.
Luckily, this is really easy with ``db.query``, you can simply add the ``query=True`` parameter to the decorator and caching will be automatically enabled!

Enabling caching for ``fetch.users`` will look like this...

.. code::

   ...

   @query(cache=True)
   def users(country: Optional[str] = None):
       ...

Now, if you execute ``fetch.users()`` it will query the database directly, but the _next_ time you execute it, it will instantaneously return the result from the previous execution.
The ``@query`` decorator will cache the results based on the query string itself, so ``fetch.users()``, ``fetch.users(country="FI")``, ``fetch.users(country="NO")``, and so on will be cached independently.

Lazy data retrieval
~~~~~~~~~~~~~~~~~~~

You can also specify the ``lazy=True`` parameter to the ``@query`` decorator in order to receive the query result in the form of a ``LazyFrame`` object rather than a ``DataFrame``.
This parameter plays well with cached query decorators since it will only read the *strictly required* data from the cache.

.. code::

   ...

   @query(cache=True, lazy=True)
   def users():
       ...

   # Only the subset of the rows with age_in_years >= 67 will be read into memory
   pensioners = users().filter(pl.col("age_in_years") >= 67).collect()


Refreshing the cache
~~~~~~~~~~~~~~~~~~~~

Sometimes you may want to forcefully reset the cache of a query function in order to get the latest version of the data from remote database.
This can be done by invoking ``X.refresh_cache()`` rather than ``X()`` directly.
Let's say you want to retrieve the latest set of Norwegian users from the database...


.. code::

   from user_analyzer import fetch

   user_df = fetch.users.refresh_cache(country="NO")

This will delete the cached version of the Norwegian users if the result has already been cached, and return the latest result.
The next time you invoke ``fetch.users(country="NO")`` you will get the latest version of the cache.
If you want to clear *all* caches, regardless query parameterization, you can use the ``X.clear_caches()`` method.

.. code::

   from user_analyzer import fetch

   fetch.users.clear_caches()

The ``.refresh_cache()`` and ``.clear_caches()`` methods are in fact part of several other methods that are automatically added to ``@query``-decorated functions, the full list of such methods is:

* ``.clear_caches()`` - Delete all cache files of the given query function such that new data will be fetched the _next_ time the query is invoked.
* ``.refresh_cache(*args, **kwargs)`` - Force the resulting SQL query produced by the given parameters to be executed in the remote database and repopulate the parameter-specific cache.
* ``.cache_path(*args, **kwargs)`` - Return a ``pathlib.Path`` object pointing to the parquet file that is used to store the cache for the given parameters.
* ``.query_string(*args, **kwargs)`` - Return the SQL query string to be executed.

Automatically refreshing old caches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it makes sense to cache a query result, but not *forever*.
In such cases you can specify the *Time to Live* (TTL) of the cache, automatically refreshing the cache when it becomes older than the specified TTL.
This can be done by specifying the ``ttl`` argument to the ``@query`` decorator as a `datetime.timedelta <https://docs.python.org/3/library/datetime.html#timedelta-objects>`_.

Let's say that we want to fetch the newest collection of users once a day, but otherwise cache the results.
This can be achieved in the following way...


.. code::

   from datetime import timedelta

   from db import query

   @query(
       cache=True,
       ttl=timedelta(days=1),
   )
   def users(country: Optional[str] = None):
       ...

The first time you invoke ``fetch.users()``, the query will be executed in the remote database and the result will be cached.
After that, the cache will be used until you invoke ``fetch.users()`` more than 24 hours after the cache was initially created.
Then the cache will be automatically refreshed.
You can also force a cache refresh any time by using the ``.refresh_cache()`` method, for instance for all Norwegian users by executing ``fetch.users.refresh_cache(country="NO")``.


Specify custom cache files (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to store the cached results in specific parquet files, you can specify the ``cache`` parameter to the ``@query`` decorator as a string or as a ``pathlib.Path`` object.
Let's say you want to store the users in a file called ``users.parquet``, this can be done in the following way:

.. code::

   from db import query

   @query(cache="users.parquet")
   def users(country: Optional[str] = None):
       ...

The file path ``users.parquet`` is a so-called *relative path* and is therefore interpreted relative the ``artifacts/query_cache`` sub-directory within the project's root.
You can inspect the resulting path by executing ``users.cache_path()``:

.. code::

   from user_analyzer import fetch

   print(fetch.users.cache_path())
   # Outputs: /repo/projects/user-analyzer/artifacts/query_cache/users.parquet

You can also specify an absolute path if required, let's say you want to place the file in ``<REPO>/projects/user-analyzer/users.parquet``:

.. code::

   from db import PROJECT_DIR, query

   @query(cache=PROJECT_DIR / "users.parquet")
   def users(country: Optional[str] = None):
       ...

The problem with the previous custom cache path is that ``fetch.users(country="NO")`` and ``fetch.users(countr="FI")`` will write to the same cache file, thus refreshing the cache much more than strictly necessary.
It would be more efficient to have a separate cache file for each country.
You can achieve this by inserting a ``{country}`` formatting placeholder, like with an f-string, in the custom cache path:

.. code::

   from db import PROJECT_DIR, query

   @query(cache=PROJECT_DIR / "users-{country}.parquet")
   def users(country: Optional[str] = None):
       ...

Finish users will now be cached in ``users-FI.parquet``, while Norwegian users will be cached in ``users-NO.parquet``.

Automatic Data Validation
-------------------------

The ``@query`` decorator integrates with the `patito <https://github.com/kolonialno/patito>`_ DataFrame validation library, allowing you to automatically validate the data fetched from the remote database.
If the concept of data validation, and why you should apply it in your data science projects, is new to you, then you should read `"Using Patito for DataFrame Validation" <https://patito.readthedocs.io/en/latest/tutorial/dataframe-validation.html>`_.

Let's say that we have a `fetch.products()` query function which produces a DataFrame of three columns.

.. code::

   from db import query

   @query()
   def products():
       return """
           select
               product_id,
               warehouse_department,
               current_retail_price

           from products
       """

Given this query we might want to validate the following assumptions:

* ``product_id`` is a unique integer assigned to each product.
* ``warehouse_department`` takes one of three permissible values: ``"Dry"``, ``"Cold"``, or ``"Frozen"``.
* ``current_retail_price`` is a positive floating point number.

By convention we should define a Patito model class named ``Product`` placed in ``<project_module>/models.py``.

.. code::

   import patito as pt


   class Product(pt.Model):
       product_id: int = pt.Field(unique=True)
       warehouse_department: Literal["Dry", "Cold", "Frozen"]
       current_retail_price: float = pt.Field(gt=0)

We can now use ``user_analyzer.models.Product`` to automatically validate the data produced by ``user_analyzer.fetch.products`` by providing the ``model`` keyword to the ``@query`` decorator.

.. code::

   from db import query

   from user_analyzer import models

   @query(model=models.Product)
   def products():
       return """
           select
               product_id,
               warehouse_department,
               current_retail_price

           from products
       """

Whenever you invoke ``fetch.products``, the data will be guaranteed to follow the schema of ``models.Product``, otherwise an exception will be raised.
You can therefore rest assured that the production data will not substantially change without you noticing it in the future.
