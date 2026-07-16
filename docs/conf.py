import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "pysamosa"
copyright = "2026, Mark Campmier"  # pylint: disable=redefined-builtin
author = "Mark Campmier, PhD"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "pysamosa"
html_theme_options = {
    "sidebar_hide_name": False,
}

myst_enable_extensions = ["colon_fence", "deflist"]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
