Using Patito for DataFrame Validation
=====================================

Have you ever found yourself relying on some column of an external data source being non-nullable only to find out `much` later that the assumption proved to be false?
What about discovering that a production machine learning model has had a huge performance regression because a new category was introduced to a categorical column?
You might not have encountered any of these `exact` scenarios, but perhaps similar ones; they illustrate the necessity of validating your data.

A machine learning model might ingest data from a production system that changes frequently, and the author of the model wants to be notified if certain assumptions no longer hold.
Or perhaps a data analyst might rely on a pre-processing step that removes all discontinued products from a data set, and this should be validated and communicated clearly in their Jupyter notebook.

`patito <https://github.com/JakobGM/patito>`_ is a dataframe validation library built on top of `polars <https://github.com/pola-rs/polars>`_ initially open sourced by Oda, which tries to solve this problem.
The polars dataframe library has lately been making the rounds among data scientists, and for good reasons.
It can be considered as a total replacement of the well-known `pandas <https://github.com/pandas-dev/pandas>`_ library, initially tempting you with its advertised `top-notch performance <https://www.pola.rs/benchmarks.html>`_, but then sealing the deal with its intuitive and expressive API.
The exact virtues of polars is a topic for another article, but suffice it to say that it is `highly` recommended and it has some great `introductory documentation <https://pola-rs.github.io/polars-book/user-guide/>`_.

The core idea of Patito is that you should define a so-called :ref:`"model" <Model>` for each of your data sources.
A `model` is a declarative python class which describes the general properties of a tabular data set: the names of all the columns, their types, value bounds, and so on...
These models can then be used to validate the data sources when they are ingested into your project's data pipeline.
In turn, your models become a trustworthy, centralized catalog of all the core facts about your data, facts you can safely rely upon during development.

Enough chit chat, let's get into some technical details!
Let's say that your project keeps track of products, and that these products have four core properties:

1. A unique, numeric identifier
2. A name
3. An ideal temperature zone of either ``"dry"``, ``"cold"``, or ``"frozen"``
4. A product demand given as a percentage of the total sales forecast for the next week

In tabular form the data might look something like this.

.. _product_table:

.. list-table:: Table 1: Products
    :widths: 25 25 25 25
    :header-rows: 1
    :align: center

    * - ``product_id``
      - ``name``
      - ``temperature_zone``
      - ``demand_percentage``
    * - 1
      - Apple
      - dry
      - 0.23%
    * - 2
      - Milk
      - cold
      - 0.61%
    * - 3
      - Ice cubes
      - frozen
      - 0.01%
    * - ...
      - ...
      - ...
      - ...

We now start to model the restrictions we want to put upon our data.
In Patito this is done by defining a class which inherits from ``patito.Model``, a class which has one `field annotation` for each column in the data.
These models should preferably be defined in a centralized place, conventionally ``<YOUR_PROJECT_NAME>/models.py``, where you can easily find and refer to them.

.. code-block:: python
   :caption: project/models.py

    from typing import Literal

    import patito as pt


    class Product(pt.Model):
        product_id: int
        name: str
        temperature_zone: Literal["dry", "cold", "frozen"]
        demand_percentage: float


Here we have used ``typing.Literal`` from `the standard library <https://docs.python.org/3/library/typing.html#typing.Literal>`_ in order to specify that ``temperature_zone`` is not only a ``str``, but `specifically` one of the literal values ``"dry"``, ``"cold"``, or ``"frozen"``.
You can now use this class to represent a `single specific instance` of a product:

.. code-block:: python

    >>> Product(product_id=1, name="Apple", temperature_zone="dry", demand_percentage=0.23)
    Product(product_id=1, name='Apple', temperature_zone='dry', demand_percentage=0.23)


The class also automatically offers input data validation, for instance if you provide an invalid value for ``temperature_zone``.

.. code-block:: python

    >>> Product(product_id=64, name="Pizza", temperature_zone="oven", demand_percentage=0.12)
    ValidationError: 1 validation error for Product
    temperature_zone
      unexpected value; permitted: 'dry', 'cold', 'frozen' (type=value_error.const; given=oven; permitted=('dry', 'cold', 'frozen'))

A discerning reader might notice that this looks suspiciously like `pydantic's <https://github.com/pydantic/pydantic>`_ data models, and that is in fact because it is!
Patito's model class is built upon pydantic's ``pydantic.BaseClass`` and therefore offers `all of pydantic's functionality <https://pydantic-docs.helpmanual.io/usage/models/>`_.
But the difference is that Patito extends pydantic's validation of `singular object instances` to `collections` of the same objects represented as `dataframes`.

Let's take the data presented in `Table 1 <product_table>`_ and represent it as a polars dataframe.

.. code-block:: python

    >>> import polars as pl

    >>> product_df = pl.DataFrame(
    ...     {
    ...         "product_id": [1, 2, 3],
    ...         "name": ["Apple", "Milk", "Ice cubes"],
    ...         "temperature_zone": ["dry", "cold", "frozen"],
    ...         "demand_percentage": [0.23, 0.61, 0.01],
    ...     }
    ... )

We can now use :ref:`Product.validate() <Model.validate>` in order to validate the content of our dataframe.

.. code-block:: python

    >>> from project.models import Product
    >>> Product.validate(product_df)
    None

Well, that wasn't really interesting...
The validate method simply returns ``None`` if no errors are found.
It is intended as a guard statement to be put before any logic that requires the data to be valid.
That way you can rely on the data being compatible with the given model schema, otherwise the ``.validate()`` method would have raised an exception.
Let's try this with invalid data, setting the temperature zone of one of the products to ``"oven"``.


.. code-block:: python

    >>> invalid_product_df = pl.DataFrame(
    ...     {
    ...         "product_id": [64, 64],
    ...         "name": ["Pizza", "Cereal"],
    ...         "temperature_zone": ["oven", "dry"],
    ...         "demand_percentage": [0.07, 0.16],
    ...     }
    ... )
    >>> Product.validate(invalid_product_df)
    ValidationError: 1 validation error for Product
    temperature_zone
      Rows with invalid values: {'oven'}. (type=value_error.rowvalue)

Now we're talking!
Patito allows you to define a single class which validates both singular object instances `and` dataframe collections without code duplication!

.. mermaid::
   :align: center

    %%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#FFF5E6', 'secondaryColor': '#FFF5E6' }}}%%
    graph LR;
        pydantic[<code class='literal'>pydantic.BaseModel</code><br /><br />Singular Instance Validation]
        patito[<code class='literal'>patito.Model</code><br /><br />Singular Instance Validation<br />+<br />DataFrame Validation]
        pydantic-->|Same class<br />definition|patito

Patito tries to rely as much as possible on pydantic's existing modelling concepts, naturally extending them to the dataframe domain where suitable.
Model fields annotated with ``str`` will map to dataframe columns stored as ``pl.Utf8``, ``int`` as ``pl.Int8``/``pl.Int16``/.../``pl.Int64``, and so on.
Field types wrapped in ``Optional`` allow null values, while bare types do not.

But certain modelling concepts are not applicable in the context of singular object instances, and are therefore necessarily not part of pydantic's API.
Take ``product_id`` as an example, you would expect this column to be unique across all products and duplicates should therefore be considered invalid.
In pydantic you have no way to express this, but Patito expands upon pydantic in various ways in order to represent dataframe-related constraints.
One of these extensions is the ``unique`` parameter accepted by ``patito.Field``, which allows you to specify that all the values of a given column should be unique.

.. code-block:: python
   :caption: project/models.py::Product

    class Product(pt.Model):
        product_id: int = pt.Field(unique=True)
        name: str
        temperature_zone: Literal["dry", "cold", "frozen"]
        demand_percentage: float


The ``patito.Field`` class accepts `the same parameters <https://pydantic-docs.helpmanual.io/usage/schema/#field-customization>`_ as ``pydantic.Field``, but adds additional dataframe-specific constraints documented :ref:`here <Field>`.
In those cases where Patito's built-in constraints do not suffice, you can specify arbitrary constraints in the form of polars `expressions <https://pola-rs.github.io/polars-book/user-guide/dsl/expressions.html>`_ which must evaluate to ``True`` for each row in order for the dataframe to be considered valid.
Let's say we want to make sure that ``demand_percentage`` sums up to 100% for the entire dataframe, otherwise we might be missing one or more products.
We can do this by passing the ``constraints`` parameter to ``patito.Field``.

.. code-block:: python
   :caption: project/models.py::Product

    class Product(pt.Model):
        product_id: int = pt.Field(unique=True)
        name: str
        temperature_zone: Literal["dry", "cold", "frozen"]
        demand_percentage: float = pt.Field(constraints=pt.field.sum() == 100.0)

Here ``patito.field`` is an alias for the field column and is automatically replaced with ``polars.col("demand_percentage")`` before validation.
If we now use this improved class to validate ``invalid_product_df``, we should detect new errors.

.. code-block:: python

    >>> Product.validate(invalid_product_df)
    ValidationError: 3 validation errors for Product
    product_id
      2 rows with duplicated values. (type=value_error.rowvalue)
    temperature_zone
      Rows with invalid values: {'oven'}. (type=value_error.rowvalue)
    demand_percentage
      2 rows does not match custom constraints. (type=value_error.rowvalue)

Patito has now detected that ``product_id`` contains duplicates and that ``demand_percentage`` does not sum up to 100%!
Several more properties and methods are available on ``patito.Model`` as outlined :ref:`here <Model>`; you can for instance generate valid mock dataframes for testing purposes with :ref:`Model.examples() <Model.examples>`.
You can also dynamically construct models with methods such as :ref:`Model.select() <Model.select>`, :ref:`Model.prefix() <Model.prefix>`, and :ref:`Model.join() <Model.join>`.
