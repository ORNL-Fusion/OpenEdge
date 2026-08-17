#!/usr/bin/env python3
"""Convert a SPARTA-format doc page (doc/*.txt) to MyST markdown that
keeps the traditional SPARTA manual anatomy (Syntax / Examples /
Description / Restrictions / Related commands).

Usage: sparta2myst.py doc/global.txt > docs/manual/global.md
"""

import re
import sys
from pathlib import Path

SECTION = re.compile(r'^\[([A-Za-z ]+):?\]$')
LISTTAG = re.compile(r'\s*:(ulb,l|ule,l|ulb|ule|l|b)\s*$')


def convert(text: str, src_name: str) -> str:
    lines = text.split('\n')
    out = []
    i = 0
    # drop the header nav block (through the first :line)
    while i < len(lines) and lines[i].strip() != ':line':
        i += 1
    i += 1

    out.append(f'<!-- Adapted from the SPARTA manual ({src_name}); '
               'Plimpton, Gallis et al., Sandia National Laboratories. '
               'OpenEdge extensions marked. -->\n')

    in_list_block = False
    para = []

    def flush_para():
        nonlocal para
        if para:
            out.append(' '.join(para))
            out.append('')
            para = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        i += 1

        if line.strip() == ':line':
            flush_para()
            out.append('---\n')
            continue
        if line.strip().startswith(':link(') or line.strip().endswith(':c'):
            continue
        m = re.match(r'^(.*)\s+:h3$', line)
        if m:
            flush_para()
            out.append(f'# {m.group(1).strip()}\n')
            continue
        m = SECTION.match(line.strip())
        if m:
            flush_para()
            if in_list_block:
                out.append('```')
                out.append('')
                in_list_block = False
            out.append(f'## {m.group(1).strip()}\n')
            continue
        # inside a keyword tree, a trailing :pre closes the whole block
        if in_list_block and line.endswith(':pre'):
            out.append(line[:-4].rstrip())
            out.append('```')
            out.append('')
            in_list_block = False
            continue
        # preformatted paragraph: PRECEDING paragraph lines + this line
        if line.endswith(':pre'):
            block = list(para)
            para = []
            block.append(line[:-4].rstrip())
            out.append('```')
            out.extend(b.strip() for b in block if b.strip())
            out.append('```')
            out.append('')
            continue
        # keyword-tree list lines: strip tags, keep as fenced block
        if LISTTAG.search(line):
            flush_para()
            if not in_list_block:
                out.append('```')
                in_list_block = True
            out.append(LISTTAG.sub('', raw.rstrip()))
            continue
        if in_list_block:
            if line.strip() == '':
                # keep the fence open across blank lines that separate
                # keyword groups; close only if the list truly ends
                j = i
                while j < len(lines) and lines[j].strip() == '':
                    j += 1
                nxt = lines[j] if j < len(lines) else ''
                if LISTTAG.search(nxt) or nxt.startswith('  '):
                    out.append('')
                else:
                    out.append('```')
                    out.append('')
                    in_list_block = False
            else:
                out.append(raw.rstrip())
            continue
        # inline cleanup for prose
        if line.strip() == '':
            flush_para()
            continue
        s = line
        s = re.sub(r'"([^"]+)"_(\S+?)(?=[\s,.)]|$)', r'\1', s)   # links -> text
        s = re.sub(r'\{(\S+?)\}', r'*\1*', s)                    # {kw} -> *kw*
        s = s.replace('[Syntax:]', '## Syntax')
        para.append(s.strip())

    flush_para()
    if in_list_block:
        out.append('```')
    # cleanup: drop empty fenced blocks and triple blank runs
    cleaned = []
    k = 0
    while k < len(out):
        if (out[k] == '```' and k + 1 < len(out) and out[k+1] == '```'):
            k += 2
            continue
        if (out[k] == '```' and k + 2 < len(out) and out[k+1] == ''
                and out[k+2] == '```'):
            k += 3
            continue
        cleaned.append(out[k])
        k += 1
    txt = '\n'.join(cleaned) + '\n'
    while '\n\n\n' in txt:
        txt = txt.replace('\n\n\n', '\n\n')
    return txt


if __name__ == '__main__':
    p = Path(sys.argv[1])
    sys.stdout.write(convert(p.read_text(), p.name))
