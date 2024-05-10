import os
import sys
from tardis_em import version

project = "TARDIS-em"
copyright = "2021-2024, Robert Kiewisz, Tristan Bepler"
author = "Robert Kiewisz, Tristan Bepler"
release = version
source_suffix = [".rst", ".md"]


sys.path.insert(0, os.path.abspath(""))

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------]

html_theme = "sphinx_rtd_theme"

autodoc_default_options = {
    "members": True,
    "private-members": False,
    "member-order": "bysource",
    "undoc-members": False,
    "inherited-members": False,
}
