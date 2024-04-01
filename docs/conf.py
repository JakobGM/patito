"""Sphinx configuration."""

import patito as pt

project = "Patito"
version = pt.__version__
author = "Jakob Gerhard Martinussen"
copyright = "2022, Oda Group Holding AS. 2023, Jakob Gerhard Martinussen & contributors"
html_theme = "sphinx_rtd_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.mermaid",
]
autodoc_member_order = "bysource"
autosummary_generate = False

# Allow the TOC tree to be expanded three levels
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 5,
}

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

html_logo = "https://em-content.zobj.net/thumbs/120/samsung/78/duck_1f986.png"
