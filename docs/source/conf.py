# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

import sys
import os
project = 'TARDIS-pytorch'
copyright = '2021, Robert Kiewisz, Tristan Bepler'
author = 'Robert Kiewisz, Tristan Bepler'
release = '0.1.0beta2'

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------]

html_theme = "furo"

autodoc_default_options = {
    'members': True,
    'private-members': False,
    'member-order': 'bysource',
    'undoc-members': False,
    'inherited-members': False,
}
