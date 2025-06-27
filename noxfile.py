"""Nox sessions.

Run with `nox -fb venv`
"""

import nox
from nox import Session
from nox_uv import session

nox.options.default_venv_backend = "uv"

nox.options.sessions = (
    "lint",
    "test",
    # "type_check",
    "docs",
)
locations = "src", "tests", "noxfile.py", "docs/conf.py"
supported_python_versions = "3.9", "3.10", "3.11", "3.12"


@session(python=supported_python_versions, uv_all_extras=True, uv_all_groups=True)
def test(session: Session):
    """Run test suite using pytest + coverage + xdoctest."""
    if session.python == "3.9":
        # Only run test coverage and docstring tests on python 3.10
        args = session.posargs  # or ["--cov", "--xdoctest"]
    else:
        args = session.posargs

    session.run("pytest", *args, external=True)


# @session(python="3.9", uv_all_extras=True, uv_all_groups=True)
# def coverage(session):
#     """Upload coverage data."""
#     install_with_constraints(session, "coverage[toml]", "codecov")
#     session.run("coverage", "xml", "--fail-under=0")
#     session.run("codecov", *session.posargs)


@session(python=["3.9", "3.13"], uv_all_extras=True, uv_all_groups=True)
def type_check(session: Session):
    """Run type-checking on project using pyright."""
    args = session.posargs or locations
    session.run("pyright", *args)
    session.run("mypy", *args, external=True)


@session(python=["3.9", "3.13"], uv_all_extras=True, uv_all_groups=True)
def lint(session: Session):
    """Run linters an project using flake8++."""
    args = session.posargs or locations
    session.run("ruff", "check", *args, external=True)
    session.run("ruff", "format", *args, external=True)


@session(python="3.9", uv_all_extras=True, uv_all_groups=True)
def format(session: Session):
    """Run the ruff formatter on the entire code base."""
    args = session.posargs or locations
    session.run("ruff format", *args, external=True)


@session(python="3.9", uv_all_extras=True, uv_all_groups=True)
def docs(session: Session) -> None:
    """Build the documentation."""
    if "--serve" in session.posargs:
        session.run("sphinx-autobuild", "docs", "docs/_build/html", external=True)
    else:
        session.run("sphinx-build", "docs", "docs/_build/html", external=True)
