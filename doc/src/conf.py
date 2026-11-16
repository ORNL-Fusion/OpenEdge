# Sphinx configuration for the OpenEdge manual (doc/src/*.rst).
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '_ext'))

project = 'OpenEdge'
author = 'Abdourahmane (Abdou) Diaw, Oak Ridge National Laboratory'
copyright = '2026, Oak Ridge National Laboratory. Built on SPARTA, Sandia Corporation'
try:
    release = subprocess.check_output(['git', 'describe', '--tags', '--always'],
                                      text=True).strip()
except Exception:
    release = 'development'
version = release

extensions = ['sphinx.ext.mathjax', 'openedge_lexer']
exclude_patterns = ['_build', '_ext', 'JPG', 'Eqs', 'Manual.rst']
master_doc = 'index'

# equations are numbered per page, e.g. (2) on a page, 'Eq. (2)' in references
math_eqref_format = '({number})'

html_theme = 'sphinx_rtd_theme'
html_title = 'OpenEdge documentation'
html_static_path = ['_static']
html_css_files = ['css/openedge.css']
pygments_style = 'default'
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'style_external_links': False,
}
html_context = {
    'display_github': True,
    'github_user': 'ORNL-Fusion',
    'github_repo': 'OpenEdge',
    'github_version': 'development',
    'conf_py_path': '/doc/src/',
}
