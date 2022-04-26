import nox

nox.options.sessions = "lint", "test"
locations = "src", "tests", "noxfile.py"


@nox.session(python=["3.9"])
def test(session):
    session.run("poetry", "install", "-E", "duckdb", external=True)
    session.run("pytest", "--cov")


@nox.session(python=["3.9"])
def lint(session):
    args = session.posargs or locations
    session.install(
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
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)
