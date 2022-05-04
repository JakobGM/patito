patito.Model
============

.. currentmodule:: patito

.. autoclass:: Model

   .. automethod:: __init__
   .. automethod:: example
   .. automethod:: example_value
   .. automethod:: examples
   .. automethod:: from_polars
   .. automethod:: from_row
   .. automethod:: pandas_examples
   .. automethod:: validate

   .. autodata:: patito.Model.__class__.valid_dtypes

   .. rubric:: Methods

   .. autosummary::

      ~Model.__init__
      ~Model.example
      ~Model.example_value
      ~Model.examples
      ~Model.from_polars
      ~Model.from_row
      ~Model.pandas_examples
      ~Model.validate

   .. rubric:: Attributes

   .. autosummary::

      ~Model.__class__.columns
      ~Model.__class__.unique_columns
      ~Model.__class__.non_nullable_columns
      ~Model.__class__.nullable_columns
      ~Model.__class__.dtypes
      ~Model.__class__.sql_types
      ~Model.__class__.valid_dtypes
      ~Model.__class__.defaults
