Overview
========

.. toctree::
   :hidden:
   :titlesonly:

   Tutorials - User Guide <tutorial/index>
   api/index
   license

Patito offers a simple way to declare pydantic data models which double as schema for your polars data frames.
These schema can be used for:

| 👮 Simple and performant data frame validation.
| 🧪 Easy generation of valid mock data frames for tests.\
| 🐍 Retrieve and represent singular rows in an object-oriented manner.\
| 🧠 Provide a single source of truth for the core data models in your code base.

Patito has first-class support for `polars <https://github.com/pola-rs/polars>`_, a *"blazingly fast DataFrames library written in Rust"*.

Installation
------------

You can simply install Patito with :code:`pip` like so:

.. code-block:: console

   pip install patito

Developer Notes
---------------

This project is managed with uv. Please install it separately.

To run tests, linting etc, please run:

.. code-block:: console

   uv run nox

To install pre-commit, please run:

.. code-block:: console

   uv run pre-commit install
