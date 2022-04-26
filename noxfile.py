import nox


@nox.session(python=["3.9"])
def tests(session):
    session.run("poetry", "install", "-E", "duckdb", external=True)
    session.run("pytest", "--cov")
