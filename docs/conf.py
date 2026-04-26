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
    "external_links": [
        {
            "name": "SPARTA Manual",
            "url": "https://sparta.github.io/doc/Manual.html",
        },
    ],
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
