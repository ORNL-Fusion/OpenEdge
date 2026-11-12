# Sphinx configuration for the OpenEdge manual (doc/src/*.rst).
import subprocess

project = 'OpenEdge'
author = 'Oak Ridge National Laboratory'
copyright = '2026, Oak Ridge National Laboratory. Built on SPARTA, Sandia Corporation'
try:
    release = subprocess.check_output(['git', 'describe', '--tags', '--always'],
                                      text=True).strip()
except Exception:
    release = 'development'
version = release

extensions = ['sphinx.ext.mathjax']
exclude_patterns = ['_build', 'JPG', 'Eqs', 'Manual.rst']
master_doc = 'index'

# equations are numbered per page, e.g. (2) on a page, 'Eq. (2)' in references
math_eqref_format = '({number})'

html_theme = 'furo'
html_title = 'OpenEdge documentation'
html_static_path = []
html_theme_options = {
    'source_repository': 'https://github.com/ORNL-Fusion/OpenEdge/',
    'source_branch': 'development',
    'source_directory': 'doc/src/',
}
