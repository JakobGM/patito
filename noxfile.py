import tempfile

import nox

nox.options.sessions = "lint", "test", "type_check"
locations = "src", "tests", "noxfile.py"
supported_python_versions = "3.8", "3.9", "3.10"


def install_with_constraints(session, *args, **kwargs):
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
    args = session.posargs or ["--cov", "--xdoctest"]
    session.run("poetry", "install", "--no-dev", "-E", "duckdb", external=True)
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
    args = session.posargs or locations
    session.run("poetry", "install", "--no-dev", "-E", "duckdb", external=True)
    install_with_constraints(session, "pyright", "pytest")
    session.run("pyright", *args)


@nox.session(python=["3.9"])
def lint(session):
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-isort",
        # TODO: And and fix all errors
        # "darglint",
        # "flake8-annotations",
        # "flake8-docstrings",
    )
    session.run("flake8", *args)


@nox.session(python="3.9")
def format(session):
    args = session.posargs or locations
    install_with_constraints(session, "black", "isort")
    session.run("black", *args)
    session.run("isort", *args)
