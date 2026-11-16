# Pygments lexer for SPARTA/OpenEdge input decks, plus a transform that
# highlights command blocks (first Syntax block, Examples, deck snippets).
from docutils import nodes
from pygments.lexer import RegexLexer, bygroups, include
from pygments.token import (Comment, Keyword, Name, Number, Operator,
                            Punctuation, String, Text, Whitespace)


class OpenEdgeLexer(RegexLexer):
    name = 'OpenEdge'
    aliases = ['openedge', 'sparta']
    filenames = ['in.*']

    tokens = {
        'root': [
            (r'\s*#.*?$', Comment.Single),
            (r'^(\s*)(\S+)', bygroups(Whitespace, Keyword), 'args'),
            (r'\s+', Whitespace),
        ],
        'args': [
            (r'\n', Whitespace, '#pop'),
            (r'&\s*\n', Punctuation),
            (r'#.*?$', Comment.Single),
            include('values'),
        ],
        'values': [
            (r'"[^"\n]*"|\'[^\'\n]*\'', String),
            (r'\$\{[^}\n]*\}|\$\w', Name.Variable),
            (r'\b[cfvs]_[\w/.\-]+(\[[^\]\s]*\])?', Name.Variable),
            (r'[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\b(?![\w/.])', Number),
            (r'\b(INF|EINF|NULL|yes|no)\b', Name.Constant),
            (r'[\w/.+\-:*]+/(kk|omp|gpu)\b', Name.Builtin),
            (r'\.\.\.', Punctuation),
            (r'[=<>!]+|\*', Operator),
            (r'[^\s"\'$#&]+', Text),
            (r'[ \t]+', Whitespace),
            (r'.', Text),
        ],
    }


def _section_title(node):
    sec = node.parent
    while sec is not None and not isinstance(sec, nodes.section):
        sec = sec.parent
    if sec is None or not sec.children or not isinstance(sec[0], nodes.title):
        return None
    return sec[0].astext().strip().lower()


def _highlight(app, doctree):
    seen_syntax = set()
    for node in list(doctree.findall(nodes.literal_block)):
        title = _section_title(node)
        lang = node.get('language')
        parsed = node.rawsource != node.astext()
        if lang == 'text' and not parsed:
            node['language'] = 'OpenEdge'
            continue
        if not parsed and lang not in (None, 'default', 'none'):
            continue
        if title == 'examples':
            pass
        elif title == 'syntax' and id(node.parent) not in seen_syntax \
                and isinstance(node.parent, nodes.section):
            seen_syntax.add(id(node.parent))
        else:
            continue
        text = node.astext()
        new = nodes.literal_block(text, text, language='OpenEdge')
        new.source, new.line = node.source, node.line
        node.replace_self(new)


def setup(app):
    app.add_lexer('OpenEdge', OpenEdgeLexer)
    app.connect('doctree-read', _highlight)
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
