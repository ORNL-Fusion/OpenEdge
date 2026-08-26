"""Sphinx configuration for the OpenEdge manual."""

from __future__ import annotations

project = "OpenEdge"
author = "OpenEdge contributors"
copyright = "2026, OpenEdge contributors"
release = "dev"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# MyST — treat Markdown as first-class, enable math + admonitions.
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
autosectionlabel_prefix_document = True

# HTML output ------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "OpenEdge"
html_short_title = "OpenEdge"

html_theme_options = {
    "github_url": "https://github.com/ORNL-Fusion/OpenEdge",
    "navigation_depth": 3,
    "show_toc_level": 2,
    "show_prev_next": True,
    "use_edit_page_button": False,
    "collapse_navigation": False,
    "navigation_with_keys": True,
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "announcement": (
        "OpenEdge is under active development — the API may still change. "
        "Check the <a href='https://github.com/ORNL-Fusion/OpenEdge/issues'>issue tracker</a> "
        "for known limitations."
    ),
}

pygments_style = "tango"

html_context = {
    "github_user": "ORNL-Fusion",
    "github_repo": "OpenEdge",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "auto",
}

html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = ["custom.css"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Copy-button behavior: skip prompt characters in shell snippets.
copybutton_prompt_text = r"^\$\s|^>>>\s|^\.\.\.\s|^In \[\d*\]:\s"
copybutton_prompt_is_regexp = True

# ---- LaTeX / PDF output ------------------------------------------------
# Build with: sphinx-build -b latex docs docs/_build/latex
#             cd docs/_build/latex && latexmk -pdf -xelatex
# Or in one shot via Makefile: cd docs && make latexpdf

latex_engine = "pdflatex"
latex_documents = [
    ("index", "OpenEdge.tex", "OpenEdge Manual",
     "OpenEdge contributors", "manual"),
]
latex_logo = None
latex_show_pagerefs = True
latex_show_urls = "footnote"
# Keep the PDF in a continuous manual style rather than splitting the
# top-level toctree captions into large "Part" divisions.
latex_toplevel_sectioning = "chapter"
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "11pt",
    "figure_align": "H",
    # Use Computer Modern fonts (always present in any TeX install) —
    # avoids TeX Gyre / FreeSerif system-font lookups.
    "fontpkg": r"\usepackage[T1]{fontenc}",
    # Extra LaTeX packages: amsmath / siunitx for physics, hyperref
    # tweaks for nicer link colors. graphicx is auto-loaded by Sphinx.
    "preamble": r"""
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\sisetup{detect-all=true}
\hypersetup{colorlinks=true, linkcolor=blue!50!black,
            urlcolor=blue!60!black, citecolor=blue!50!black}
""",
    "extraclassoptions": "openany,oneside",
}
