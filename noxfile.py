"""Nox sessions."""
import tempfile

import nox  # type: ignore

nox.options.sessions = "lint", "test", "type_check"
locations = "src", "tests", "noxfile.py", "docs/conf.py"
supported_python_versions = "3.7", "3.8", "3.9", "3.10", "3.11"


def install_with_constraints(session, *args, **kwargs):
    """
    Install packages constrained by Poetry's lock file.

    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.

    Args:
        session: The Session object.
        *args: Command-line arguments for pip.
        **kwargs: Additional keyword arguments for Session.install.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--without-hashes",
            "--with=dev",
            "--format=constraints.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=supported_python_versions)
def test(session):
    """Run test suite using pytest + coverage + xdoctest."""
    if session.python == "3.9":
        # Only run test coverage and docstring tests on python 3.10
        args = session.posargs or ["--cov", "--xdoctest"]
    else:
        args = session.posargs

    session.run(
        "poetry",
        "install",
        "--only=main",
        "--extras",
        # Pandas/pyarrow requires python version >= 3.8
        "duckdb" if session.python == "3.7" else "caching duckdb pandas",
        external=True,
    )
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
        "xdoctest",
    )
    session.run("pytest", *args)


@nox.session(python="3.9")
def coverage(session):
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


@nox.session(python=["3.11"])
def type_check(session):
    """Run type-checking on project using pyright."""
    args = session.posargs or locations
    session.run(
        "poetry",
        "install",
        "--only=main",
        "--extras",
        "caching duckdb pandas",
        external=True,
    )
    install_with_constraints(
        session, "mypy", "pyright", "pytest", "types-setuptools", "pandas-stubs"
    )
    session.run("pyright", *args)
    session.run("mypy", *args)


@nox.session(python=["3.11"])
def lint(session):
    """Run linters an project using flake8++."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-isort",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python="3.11")
def format(session):
    """Run the black formatter on the entire code base."""
    args = session.posargs or locations
    install_with_constraints(session, "black", "isort")
    session.run("black", *args)
    session.run("isort", *args)


@nox.session(python="3.9")
def docs(session) -> None:
    """Build the documentation."""
    session.run(
        "poetry",
        "install",
        "--only=main",
        "--extras",
        "caching duckdb pandas",
        external=True,
    )
    install_with_constraints(
        session,
        "sphinx",
        "sphinx-autodoc-typehints",
        "sphinx-rtd-theme",
        "sphinx-autobuild",
    )
    if "--serve" in session.posargs:
        session.run("sphinx-autobuild", "docs", "docs/_build/html")
    else:
        session.run("sphinx-build", "docs", "docs/_build/html")
