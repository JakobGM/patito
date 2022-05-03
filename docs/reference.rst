Reference
=========

.. contents::
    :local:
    :backlinks: none

.. autosummary::
   patito.Model
   patito.DataFrame

.. autoclass:: patito.Model
   :members:

   .. TODO: Remove once we drop support for python 3.6
   .. and can represent these with @property + @classmethod
   .. autodata:: patito.Model.__class__.columns
   .. autodata:: patito.Model.__class__.unique_columns
   .. autodata:: patito.Model.__class__.non_nullable_columns
   .. autodata:: patito.Model.__class__.nullable_columns
   .. autodata:: patito.Model.__class__.dtypes
   .. autodata:: patito.Model.__class__.sql_types
   .. autodata:: patito.Model.__class__.valid_dtypes
   .. autodata:: patito.Model.__class__.defaults

.. autoclass:: patito.DataFrame
   :members:
