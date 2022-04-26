import nox

nox.options.sessions = "lint", "tests"
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
        "flake8-black",
        "flake8-isort",
    )
    session.run("flake8", *args)


@nox.session(python="3.9")
def format(session):
    args = session.posargs or locations
    session.install("black", "isort")
    session.run("black", *args)
    session.run("isort", *args)
