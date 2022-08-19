"""Sphinx configuration."""
project = "Patito"
author = "Jakob Gerhard Martinussen"
copyright = "2022, Oda Group Holding AS"
html_theme = "sphinx_rtd_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]
autodoc_member_order = "bysource"
autosummary_generate = False

# Allow the TOC tree to be expanded three levels
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
