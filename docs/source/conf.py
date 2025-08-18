# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LLM-GE'
copyright = '2025, Clint Morris, Jason Zutty, et al.'
author = 'Clint Morris, Jason Zutty, et al.'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosummary',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode']

autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

# -- Ensure that the code appears in the path
import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', '..', 'src', 'utils').resolve()))
sys.path.insert(0, str(Path('..', '..', 'src', 'cfg').resolve()))
sys.path.insert(0, str(Path('..', '..', 'src').resolve()))
sys.path.insert(0, str(Path('..', '..').resolve()))
print(sys.path)
# apidoc_modules = [
#     {'path': '../../run_imporved.py', 'destination': 'source/'},
#     {'path': '../../src/llm_crossover.py', 'destination': 'source/'},
#     {'path': '../../src/llm_mutation.py', 'destination': 'source/'},
#     {'path': '../../src/llm_utils.py', 'destination': 'source/'},
#     {'path': '../../src/cfg/constants.py', 'destination': 'source/'},
#     {'path': '../../src/utils/print_utils.py', 'destination': 'source/'}
# ]