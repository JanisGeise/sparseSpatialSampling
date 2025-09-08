# -- Path setup --------------------------------------------------------------
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('../..'))  # allow "import sparseSpatialSampling"


# -- Project information -----------------------------------------------------
project = 'Sparse Spatial Sampling'
author = 'sparse spatial smapling contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # Google/NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]
autosummary_generate = True

# If your code imports heavy/optional deps not installed during docs build,
# list them here to avoid import errors while autodoc runs:
autodoc_mock_imports = [
]

# Exclude tests and any junk from the docs build
exclude_patterns = [
    'build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints',
    '**/sparseSpatialSampling/tests/*', 'setup*', '**/sparseSpatialSampling/version*'
]
autoclass_content = "both"

# Sphinx 5+ prefers root_doc (index by default). Keep explicit:
root_doc = 'index'

# Make API pages a bit richer by default:
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
