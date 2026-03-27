import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "SpikeLab"
copyright = "2025, SpikeLab Maintainers"
author = "SpikeLab Maintainers"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

# Autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autosummary_generate = True

# Mock optional dependencies so docs build without them
autodoc_mock_imports = [
    "boto3",
    "numba",
    "sklearn",
    "umap",
    "networkx",
    "community",
    "neo",
    "quantities",
    "pynwb",
    "jax",
    "jaxlib",
    "jaxopt",
    "optax",
    "poor_man_gplvm",
]

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
