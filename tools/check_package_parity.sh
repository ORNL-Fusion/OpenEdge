#!/bin/bash
# Package parity gate: src/ is the authoritative tree, src/OPENEDGE/ is a
# verbatim mirror of every OpenEdge-owned file, and Install.sh lists each one.
#
#   tools/check_package_parity.sh            check the working tree
#   tools/check_package_parity.sh --install  also run `make yes-openedge` on a
#                                            scratch copy and require no change
#
# Prints one line per finding and exits 1 if there is any:
#   MISMATCH f     src/f and src/OPENEDGE/f differ
#   PKG-ONLY f     package file with no root twin
#   ROOT-ONLY f    root file with an OpenEdge header marker and no package twin
#   STALE-ENTRY f  Install.sh names a file that is not in src/OPENEDGE
#   UNLISTED f     src/OPENEDGE file missing from Install.sh
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/src; pkg=$src/OPENEDGE; inst=$pkg/Install.sh
findings=0
say() { echo "$@"; findings=$((findings+1)); }

for f in "$pkg"/*.cpp "$pkg"/*.h; do
  b=$(basename "$f")
  if [ -e "$src/$b" ]; then cmp -s "$f" "$src/$b" || say "MISMATCH $b"
  else say "PKG-ONLY $b"; fi
done
for f in "$src"/*.cpp "$src"/*.h; do
  b=$(basename "$f")
  [ -e "$pkg/$b" ] && continue
  head -12 "$f" | grep -q "OpenEdge" && say "ROOT-ONLY $b"
done
grep -E '^(override|action) [A-Za-z0-9_.]+$' "$inst" | awk '{print $2}' | while read -r b; do
  [ -e "$pkg/$b" ] || echo "STALE-ENTRY $b"
done | while read -r line; do say "$line"; done
for f in "$pkg"/*.cpp "$pkg"/*.h; do
  b=$(basename "$f")
  grep -qE "^(override|action) $b\$" "$inst" || say "UNLISTED $b"
done
# subshell pipes above cannot bump the counter: recount from a second pass
stale=$(grep -E '^(override|action) [A-Za-z0-9_.]+$' "$inst" | awk '{print $2}' | while read -r b; do [ -e "$pkg/$b" ] || echo x; done | wc -l)
findings=$((findings+stale))

if [ "${1:-}" = "--install" ]; then
  tmp=$(mktemp -d); cp -r "$src" "$tmp/src"
  ( cd "$tmp/src" && make yes-openedge >/dev/null 2>&1 ) || say "INSTALL make yes-openedge failed"
  diff -rq "$src" "$tmp/src" -x 'Makefile.package*' -x '*.sparta_orig' -x '*.o' -x 'Obj_*' \
    | sed 's/^/INSTALL /' | while read -r line; do echo "$line"; done
  changed=$(diff -rq "$src" "$tmp/src" -x 'Makefile.package*' -x '*.sparta_orig' -x '*.o' -x 'Obj_*' | wc -l)
  findings=$((findings+changed))
  rm -rf "$tmp"
fi

if [ "$findings" -gt 0 ]; then echo "package parity: $findings finding(s)"; exit 1; fi
echo "package parity: ok"
