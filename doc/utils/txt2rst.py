#!/usr/bin/env python3
"""Convert OpenEdge/SPARTA-style doc/*.txt pages to reStructuredText.

Paragraphs end at a blank line or at a line whose last token is a format
command (":pre", ":ulb,l", ":link(x),h4", ...).  Inline markup: [bold],
{italic}, "text"_target.  Only the subset of the SPARTA txt2html markup
used in doc/*.txt is handled.

Pages whose .rst starts with the HAND_EDITED marker are maintained
directly in doc/src and are never overwritten.

    python3 doc/utils/txt2rst.py -o doc/src doc/*.txt
"""
import argparse
import os
import re
import sys

CMD_RE = re.compile(r'(?:^|\s):([a-zA-Z][a-zA-Z0-9]*(?:\([^)]*\))?'
                    r'(?:,[a-zA-Z][a-zA-Z0-9]*(?:\([^)]*\))?)*)\s*$')
LINK_RE = re.compile(r'"([^"]+)"_(\S+?)(?=[,.;:?!)]*(?:\s|$))')
HAND_EDITED = '.. hand-edited'
HEADER_UNDERLINE = {1: '#', 2: '*', 3: '=', 4: '-', 5: '^', 6: '"'}


def split_commands(spec):
    """':link(a,b),h4' -> [('link', 'a,b'), ('h4', None)]"""
    out = []
    for m in re.finditer(r'([a-zA-Z][a-zA-Z0-9]*)(?:\(([^)]*)\))?', spec):
        out.append((m.group(1), m.group(2)))
    return out


def split_paragraphs(text):
    """Yield (text, [(cmd, arg), ...]) for every paragraph."""
    lines, paras = [], []

    def flush(cmds):
        body = '\n'.join(lines).rstrip('\n')
        if body.strip() or cmds:
            paras.append((body, cmds))
        lines.clear()

    for raw in text.splitlines():
        if not raw.strip():
            flush([])
            continue
        m = CMD_RE.search(raw)
        if m:
            lines.append(raw[:m.start(1) - 1].rstrip())
            flush(split_commands(m.group(1)))
        else:
            lines.append(raw.rstrip())
    flush([])
    return paras


class Converter:
    def __init__(self, aliases):
        self.aliases = aliases          # name -> url/page, from :link(name,url)
        self.indent = 0                 # list nesting depth (ulb/ule)
        self.pending_term = None        # :dt waiting for its :dd
        self.level = 3                  # last header level seen
        self.out = []

    # ---- inline markup -------------------------------------------------

    def link(self, text, target):
        text = ' '.join(text.split())
        target = self.aliases.get(target, target)
        if target.startswith(('http://', 'https://', 'mailto:')):
            return '`%s <%s>`_' % (text, target)
        if '#' in target:
            return ':ref:`%s <%s>`' % (text, target.split('#', 1)[1])
        if target.endswith('.html'):
            return ':doc:`%s <%s>`' % (text, target[:-5])
        return '`%s <%s>`_' % (text, target)

    def inline(self, text):
        holes = []

        def stash(s):
            holes.append(s)
            return '\x00%d\x00' % (len(holes) - 1)

        text = LINK_RE.sub(lambda m: stash(self.link(m.group(1), m.group(2))), text)
        # literal brackets/braces written as \[ \] \{ \}
        text = re.sub(r'\\([\[\]{}])', lambda m: stash(m.group(1)), text)
        # RST specials
        text = text.replace('\\', '\\\\')
        text = text.replace('*', '\\*').replace('`', '\\`').replace('|', '\\|')
        text = re.sub(r'(\w)_(?=\W|$)', r'\1\\_', text)
        text = re.sub(r'(^|\s)_(?=\w)', r'\1\\_', text)
        # [bold] and {italic}
        text = re.sub(r'\[([^\]]+)\]', lambda m: stash('**%s**' % m.group(1)), text)
        text = re.sub(r'\{([^}]+)\}', lambda m: stash('*%s*' % m.group(1)), text)
        # inline markup needs a boundary; add an escaped space where a word touches it
        text = re.sub(r'(\w)(\x00(\d+)\x00)',
                      lambda m: m.group(1) + ('\\ ' if holes[int(m.group(3))][0] in '*`:' else '') + m.group(2), text)
        text = re.sub(r'(\x00(\d+)\x00)(\w)',
                      lambda m: m.group(1) + ('\\ ' if holes[int(m.group(2))][0] in '*`:' else '') + m.group(3), text)
        while '\x00' in text:
            text = re.sub(r'\x00(\d+)\x00', lambda m: holes[int(m.group(1))], text)
        return text

    # ---- block helpers -------------------------------------------------

    def emit(self, block, item=False):
        pad = '  ' * max(0, self.indent - (1 if item else 0))
        self.out.append('\n'.join(pad + l if l else '' for l in block.split('\n')))
        self.out.append('')

    def header(self, text, level):
        text = ' '.join(self.inline(text).split())
        if level == 3:
            m = re.match(r'(.+?) command$', text)
            if m:
                self.out.append('.. index:: %s\n' % m.group(1))
        self.level = level
        self.emit(text + '\n' + HEADER_UNDERLINE[level] * len(text))

    def preformatted(self, text):
        body = '\n'.join('   ' + self.inline(l) for l in text.split('\n'))
        self.emit('.. parsed-literal::\n\n' + body)

    def bullets(self, text, marker='*'):
        self.emit('\n'.join('%s %s' % (marker, self.inline(l)) for l in text.split('\n')))

    def list_item(self, text):
        body = self.inline(text).split('\n')
        self.emit('* ' + body[0] + ''.join('\n  ' + l for l in body[1:]), item=True)

    def line_block(self, text):
        self.emit('\n'.join('| ' + self.inline(l) for l in text.split('\n')))

    def table(self, text, spec):
        opts = dict(kv.split('=', 1) for kv in (spec or '').split(',') if '=' in kv)
        sep = opts.get('s', ',')
        ncol = int(opts.get('c', 0))
        if ncol:
            cells = [c.strip() for c in text.replace('\n', ' ').split(sep)]
            rows = [cells[i:i + ncol] for i in range(0, len(cells), ncol)]
        else:
            rows = [[c.strip() for c in l.split(sep)] for l in text.split('\n')]
        ncol = max(len(r) for r in rows)
        body = []
        for r in rows:
            r = r + [''] * (ncol - len(r))
            body.append('   * - ' + self.inline(r[0]))
            body += ['     - ' + self.inline(c) for c in r[1:]]
        self.emit('.. list-table::\n\n' + '\n'.join(body))

    def image(self, arg, center):
        parts = [p.strip() for p in arg.split(',')]
        block = '.. image:: ' + parts[0]
        if len(parts) > 1:
            block += '\n   :target: ' + parts[1]
        if center:
            block += '\n   :align: center'
        self.emit(block)

    def paragraph(self, text):
        m = re.match(r'^\[([^\]]+:)\]$', text.strip())
        if m:                                   # [Syntax:] style sub-heading
            title = m.group(1).rstrip(':')
            self.emit(title + '\n' + HEADER_UNDERLINE[min(6, self.level + 1)] * len(title))
            return
        m = re.match(r'^(IMPORTANT NOTE|NOTE):\s+(.*)$', text, re.S)
        if m:
            kind = 'warning' if m.group(1).startswith('IMPORTANT') else 'note'
            body = '\n'.join('   ' + l for l in self.inline(m.group(2)).split('\n'))
            self.emit('.. %s::\n\n%s' % (kind, body))
            return
        self.emit(self.inline(text))

    # ---- driver --------------------------------------------------------

    def convert(self, paras):
        started = False
        for text, cmds in paras:
            names = [c for c, _ in cmds]
            if not started:                     # drop nav header up to first :line
                if 'line' in names:
                    started = True
                continue
            if re.match(r'^(\s*<[^>]+>\s*)+$', text):   # stray raw HTML tags
                continue
            if 'ulb' in names or 'olb' in names or 'dlb' in names:
                self.indent += 1 if 'ulb' in names or 'olb' in names else 0
            for cmd, arg in cmds:
                if cmd == 'link' and ',' not in (arg or ''):
                    self.out.append('.. _%s:\n' % arg)
            # body
            center = 'c' in names
            if 'line' in names:
                self.emit('----------')
            elif any(n.startswith('h') and n[1:].isdigit() for n in names):
                lvl = int([n for n in names if n[0] == 'h' and n[1:].isdigit()][0][1:])
                self.header(text, lvl)
            elif 'image' in names:
                for cmd, arg in cmds:
                    if cmd == 'image':
                        self.image(arg, center)
                if text.strip():
                    self.paragraph(text)
            elif 'pre' in names:
                self.preformatted(text)
            elif 'tb' in names:
                self.table(text, dict(cmds)['tb'])
            elif 'ul' in names:
                self.bullets(text)
            elif 'ol' in names:
                self.bullets(text, '#.')
            elif 'l' in names:
                self.list_item(text)
            elif 'dt' in names:
                self.pending_term = ' '.join(self.inline(text).split())
            elif 'dd' in names:
                body = '\n'.join('   ' + l for l in self.inline(text).split('\n'))
                self.emit('%s\n%s' % (self.pending_term or '', body))
                self.pending_term = None
            elif 'all' in names:
                self.line_block(text)
            elif text.strip():
                self.paragraph(text)
            if 'ule' in names or 'ole' in names:
                self.indent = max(0, self.indent - 1)
        result = '\n'.join(self.out).rstrip() + '\n'
        return re.sub(r'(----------\n\n)+(?=----------\n)', '', result)


def convert_file(path):
    text = open(path, encoding='utf-8').read()
    paras = split_paragraphs(text)
    aliases = {}
    for _, cmds in paras:
        for cmd, arg in cmds:
            if cmd == 'link' and arg and ',' in arg:
                name, url = arg.split(',', 1)
                aliases[name.strip()] = url.strip()
    return Converter(aliases).convert(paras)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('-o', '--output-dir', default='.')
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for path in args.files:
        base = os.path.splitext(os.path.basename(path))[0]
        dest = os.path.join(args.output_dir, base + '.rst')
        if os.path.exists(dest):
            with open(dest, encoding='utf-8') as f:
                if f.readline().startswith(HAND_EDITED):
                    continue
        try:
            rst = convert_file(path)
        except Exception as e:
            print('%s: %s' % (path, e), file=sys.stderr)
            continue
        with open(dest, 'w', encoding='utf-8') as f:
            f.write(rst)


if __name__ == '__main__':
    main()
