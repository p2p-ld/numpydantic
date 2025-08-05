# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata as metadata

project = "numpydantic"
copyright = "2024, Jonny Saunders"
author = "Jonny Saunders"
release = metadata.version("numpydantic")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "myst_nb",
    "sphinx.ext.todo",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "linkml": ("https://linkml.io/linkml/", None),
    "linkml_runtime": ("https://linkml.io/linkml/", None),
    "linkml-runtime": ("https://linkml.io/linkml/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/notebooks.css"]

# autodoc
autodoc_pydantic_model_show_json_error_strategy = "coerce"
autodoc_pydantic_model_show_json = False

autoclass_content = "both"
autodoc_member_order = "bysource"
add_module_names = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# todo
todo_include_todos = True
todo_link_only = True

# myst
# myst-nb
nb_render_markdown_format = "myst"
nb_execution_show_tb = True
exclude_patterns = ["_build", "jupyter_execute"]
