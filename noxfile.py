"""Nox sessions."""
import tempfile

import nox

nox.options.sessions = "lint", "test", "type_check"
locations = "src", "tests", "noxfile.py"
supported_python_versions = "3.7", "3.8", "3.9", "3.10"


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
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=supported_python_versions)
def test(session):
    """Run test suite using pytest + coverage + xdoctest."""
    if session.python == "3.10":
        # Only run test coverage and docstring tests on python 3.10
        args = session.posargs or ["--cov", "--xdoctest"]
    else:
        args = session.posargs

    session.run(
        "poetry",
        "install",
        "--no-dev",
        "--extras",
        # Pandas requires python version >= 3.8
        "duckdb" if session.python == "3.7" else "duckdb pandas",
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


@nox.session(python=["3.9"])
def type_check(session):
    """Run type-checking on project using pyright."""
    args = session.posargs or locations
    session.run(
        "poetry",
        "install",
        "--no-dev",
        "--extras",
        "duckdb pandas",
        external=True,
    )
    install_with_constraints(session, "pyright", "pytest")
    session.run("pyright", *args)


@nox.session(python=["3.9"])
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


@nox.session(python="3.9")
def format(session):
    """Run the black formatter on the entire code base."""
    args = session.posargs or locations
    install_with_constraints(session, "black", "isort")
    session.run("black", *args)
    session.run("isort", *args)
