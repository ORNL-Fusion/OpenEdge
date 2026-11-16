"""Static reader of SPARTA/OpenEdge input decks.

Follows the input rules of the code: '&' continuation, '#' comments outside
quotes, $-substitution outside quotes, index-style variables not redefined,
include, if/then/elif/else, jump SELF <label>.  Anything it cannot evaluate is
recorded in Deck.unknown and the commands it affects are marked `conditional`,
so later checks report "unknown" instead of "pass" (fail closed).
"""
import ast
import math
import operator
import os
import re

INDEX_STYLES = {"index", "loop", "world", "universe", "uloop"}


class Unknown(Exception):
    """A value that cannot be determined statically."""


class Command:
    __slots__ = ("name", "words", "file", "line", "conditional", "raw")

    def __init__(self, name, words, file, line, conditional, raw):
        self.name, self.words, self.file, self.line = name, words, file, line
        self.conditional, self.raw = conditional, raw

    @property
    def where(self):
        return "%s:%d" % (os.path.basename(self.file), self.line)

    def as_dict(self):
        return {"name": self.name, "args": self.words, "where": self.where,
                "conditional": self.conditional}


class Deck:
    def __init__(self):
        self.commands = []
        self.variables = {}        # name -> (style, values)
        self.unknown = []          # {"where", "what"}
        self.files = []            # input files read (deck + includes)


# ---------------------------------------------------------------- expressions

_FUNCS = {"sqrt": math.sqrt, "exp": math.exp, "ln": math.log, "log": math.log10,
          "abs": abs, "ceil": math.ceil, "floor": math.floor, "round": round,
          "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan": math.atan,
          "asin": math.asin, "acos": math.acos, "atan2": math.atan2,
          "min": min, "max": max}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
        ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}


def _translate(expr):
    e = expr.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = e.replace("^", "**")
    return e


def evaluate(expr, lookup):
    """Evaluate a SPARTA equal-style/if expression. lookup(name) -> value."""
    try:
        tree = ast.parse(_translate(expr.strip()), mode="eval")
    except SyntaxError:
        # bare string comparisons: total == total
        m = re.fullmatch(r"\s*(\S+)\s*(==|!=)\s*(\S+)\s*", expr)
        if m:
            a, op, b = m.groups()
            return (a == b) if op == "==" else (a != b)
        raise Unknown("cannot parse expression '%s'" % expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return v
            if isinstance(n.op, ast.Not):
                return not v
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, c in zip(n.ops, n.comparators):
                right = ev(c)
                if type(op) not in _CMP or not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, ast.Name):
            if n.id.startswith("v_"):
                return _num(lookup(n.id[2:]))
            # a bare word compared as a string (if "${mode} == total")
            return n.id
        raise Unknown("unsupported expression '%s'" % expr)

    return ev(tree)


def _num(v):
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in str(v) and "e" not in str(v).lower() else f
    except (TypeError, ValueError):
        return v


def fmt_number(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v) if not v.is_integer() or abs(v) >= 1e16 else str(int(v))
    return str(v)


# ---------------------------------------------------------------- lexing

def split_words(line):
    """Split on whitespace; quoted strings ('', "", \"\"\") are one word, quotes removed."""
    words, i, n = [], 0, len(line)
    while i < n:
        if line[i].isspace():
            i += 1
            continue
        if line.startswith('"""', i):
            j = line.find('"""', i + 3)
            if j < 0:
                raise Unknown("unterminated triple quote")
            words.append(line[i + 3:j])
            i = j + 3
        elif line[i] in "\"'":
            q = line[i]
            j = line.find(q, i + 1)
            if j < 0:
                raise Unknown("unterminated quote")
            words.append(line[i + 1:j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace() and line[j] not in "\"'":
                j += 1
            words.append(line[i:j])
            i = j
    return words


def strip_comment(line):
    out, q, i = [], None, 0
    while i < len(line):
        if q is None and line.startswith('"""', i):
            q = '"""'
            out.append(q)
            i += 3
            continue
        if q == '"""' and line.startswith('"""', i):
            q = None
            out.append('"""')
            i += 3
            continue
        c = line[i]
        if q is None and c == "#":
            break
        if q is None and c in "\"'":
            q = c
        elif q == c:
            q = None
        out.append(c)
        i += 1
    return "".join(out)


def logical_lines(text):
    """Yield (line_number, text) with '&' continuations joined and comments removed."""
    buf, start = "", None
    for k, raw in enumerate(text.splitlines(), 1):
        s = strip_comment(raw).rstrip()
        if start is None:
            start = k
        if s.endswith("&"):
            buf += s[:-1] + " "
            continue
        buf += s
        if buf.strip():
            yield start, buf
        buf, start = "", None
    if buf.strip():
        yield start, buf


# ---------------------------------------------------------------- reader

class Reader:
    def __init__(self, run_dir, overrides=None, env=None):
        self.run_dir = run_dir
        self.deck = Deck()
        self.env = env if env is not None else os.environ
        for k, v in (overrides or {}).items():
            self.deck.variables[k] = ("index", [str(v)])

    # variable value as a string, raising Unknown
    def value(self, name):
        if name not in self.deck.variables:
            raise Unknown("variable '%s' is not defined" % name)
        style, vals = self.deck.variables[name]
        if style in INDEX_STYLES or style == "string":
            return vals[0]
        if style == "equal":
            try:
                v = evaluate(" ".join(vals), self.value)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
                raise Unknown("variable '%s' not evaluated: %s" % (name, e))
            if isinstance(v, str):
                raise Unknown("variable '%s' depends on a run-time value ('%s')" % (name, v))
            return fmt_number(v)
        if style == "getenv":
            if vals[0] in self.env:
                return self.env[vals[0]]
            raise Unknown("environment variable '%s' not set" % vals[0])
        raise Unknown("variable '%s' has style '%s'" % (name, style))

    def substitute(self, line):
        """$-substitution outside quotes, as the code does at read time."""
        out, i, q = [], 0, None
        while i < len(line):
            c = line[i]
            if q is None and line.startswith('"""', i):
                j = line.find('"""', i + 3)
                j = len(line) if j < 0 else j + 3
                out.append(line[i:j])
                i = j
                continue
            if q is None and c in "\"'":
                q = c
            elif q == c:
                q = None
            if c == "$" and q is None and i + 1 < len(line):
                if line[i + 1] == "{":
                    j = line.index("}", i)
                    out.append(self.value(line[i + 2:j]))
                    i = j + 1
                    continue
                if line[i + 1] == "(":
                    depth, j = 0, i + 1
                    while j < len(line):
                        depth += line[j] == "("
                        depth -= line[j] == ")"
                        if depth == 0:
                            break
                        j += 1
                    try:
                        v = evaluate(line[i + 2:j], self.value)
                    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as e:
                        raise Unknown("expression '%s' not evaluated: %s" % (line[i + 2:j], e))
                    if isinstance(v, str):
                        raise Unknown("expression '%s' depends on a run-time value" % line[i + 2:j])
                    out.append(fmt_number(v))
                    i = j + 1
                    continue
                out.append(self.value(line[i + 1]))
                i += 2
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def note(self, where, what):
        self.deck.unknown.append({"where": where, "what": what})

    def read(self, path, conditional=False):
        path = path if os.path.isabs(path) else os.path.join(self.run_dir, path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.deck.files.append(os.path.abspath(path))
        with open(path, errors="replace") as f:
            lines = list(logical_lines(f.read()))
        self._exec_lines(lines, path, conditional)
        return self.deck

    def _exec_lines(self, lines, path, conditional):
        k = 0
        while k < len(lines):
            lineno, text = lines[k]
            k += 1
            jump = self.execute(text, path, lineno, conditional)
            if jump is None:
                continue
            target, cond = jump
            idx = next((m for m in range(k, len(lines))
                        if split_words(lines[m][1])[:2] == ["label", target]), None)
            if idx is None:
                self.note("%s:%d" % (os.path.basename(path), lineno),
                          "jump to label '%s' that is not later in the file (loop?): "
                          "rest of the file read as conditional" % target)
                conditional = True
                continue
            if cond:
                # unresolved jump: read the skipped lines as conditional
                for m in range(k, idx):
                    self.execute(lines[m][1], path, lines[m][0], True)
            k = idx

    def execute(self, text, path, lineno, conditional):
        where = "%s:%d" % (os.path.basename(path), lineno)
        try:
            line = self.substitute(text)
            words = split_words(line)
        except (Unknown, ValueError, TypeError) as e:
            self.note(where, str(e))
            words = split_words_safe(text)
            if not words:
                return None
            self.deck.commands.append(Command(words[0], words[1:], path, lineno, True, text))
            return None
        if not words:
            return None
        name, args = words[0], words[1:]
        self.deck.commands.append(Command(name, args, path, lineno, conditional, text))

        if name == "variable" and len(args) >= 2:
            vname, style = args[0], args[1]
            if style == "delete":
                self.deck.variables.pop(vname, None)
            elif style in INDEX_STYLES and vname in self.deck.variables:
                pass  # index-style variables are not redefined (command line wins)
            else:
                self.deck.variables[vname] = (style, args[2:])
        elif name == "include" and args:
            try:
                self.read(args[0], conditional)
            except FileNotFoundError:
                self.note(where, "include file '%s' not found" % args[0])
        elif name == "jump" and len(args) >= 2 and args[0] == "SELF":
            return (args[1], conditional)
        elif name == "jump":
            self.note(where, "jump to another file is not followed")
        elif name == "if":
            return self._if(args, path, lineno, where, conditional)
        elif name == "next":
            self.note(where, "loop variables ('next') are not followed")
        return None

    def _if(self, args, path, lineno, where, conditional):
        # if "cond" then "c1" "c2" ... elif "cond" "c" ... else "c" ...
        branches, i = [], 0
        cond, cmds = args[0] if args else "", []
        i = 1
        if i < len(args) and args[i] == "then":
            i += 1
        while i < len(args):
            if args[i] == "elif" and i + 1 < len(args):
                branches.append((cond, cmds))
                cond, cmds = args[i + 1], []
                i += 2
            elif args[i] == "else":
                branches.append((cond, cmds))
                cond, cmds = None, []
                i += 1
            else:
                cmds.append(args[i])
                i += 1
        branches.append((cond, cmds))
        chosen, unresolved = None, False
        for cond, cmds in branches:
            if cond is None:
                chosen = cmds
                break
            try:
                if evaluate(self.substitute(cond), self.value):
                    chosen = cmds
                    break
            except (Unknown, ValueError, ZeroDivisionError, TypeError) as e:
                self.note(where, "if condition '%s' not evaluated (%s): all branches "
                          "read as conditional" % (cond, e))
                unresolved = True
                break
        if unresolved:
            jump = None
            for _, cmds in branches:
                for c in cmds:
                    j = self.execute(c, path, lineno, True)
                    jump = jump or (j and (j[0], True))
            return jump
        for c in chosen or []:
            jump = self.execute(c, path, lineno, conditional)
            if jump is not None:
                return jump  # commands after a jump in the same if are not run
        return None


def split_words_safe(text):
    try:
        return split_words(text)
    except Unknown:
        return text.split()


def read_deck(path, overrides=None, run_dir=None, env=None):
    """Read a deck the way the code would when started in run_dir (default: deck dir)."""
    path = os.path.abspath(path)
    run_dir = os.path.abspath(run_dir or os.path.dirname(path))
    r = Reader(run_dir, overrides, env)
    deck = r.read(path)
    deck.run_dir = run_dir
    deck.path = path
    deck.overrides = dict(overrides or {})
    return deck
