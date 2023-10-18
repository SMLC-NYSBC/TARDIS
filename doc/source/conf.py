import os
import sys
from tardis_em import version

project = "TARDIS-em"
copyright = "2021, Robert Kiewisz, Tristan Bepler"
author = "Robert Kiewisz, Tristan Bepler"
release = version

sys.path.insert(0, os.path.abspath(""))

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------]

html_theme = "furo"

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "member-order": "bysource",
    "undoc-members": False,
    "inherited-members": False,
}
