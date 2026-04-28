#!/bin/bash
# Download large test data files for OpenEdge examples.
#
# Plasma, B-field, equilibrium, and geometry files are too large for the
# git repository. They are distributed as per-test tarballs attached to
# a GitHub release.
#
# Usage:
#   ./download_data.sh                 # download all test data
#   ./download_data.sh test_west_3d    # download one test
#   ./download_data.sh --list          # list available datasets

set -euo pipefail
trap 'rm -f /tmp/openedge-testdata-*.tar.gz' EXIT

REPO="ORNL-Fusion/OpenEdge"
VERSION="latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASETS=(
    test_west_3d
    test_evaporation
    test_west_axi
    test_d3d_mateja
    test_d3d_walldyn
    test_gca
)

usage() {
    echo "Usage: $0 [--list] [--version TAG] [TESTNAME ...]"
    echo ""
    echo "Download test data for OpenEdge examples."
    echo ""
    echo "Options:"
    echo "  --list          List available datasets"
    echo "  --version TAG   Download from a specific release (default: latest)"
    echo ""
    echo "Examples:"
    echo "  $0                          # download all"
    echo "  $0 test_west_3d test_gca    # download specific tests"
    exit 0
}

list_datasets() {
    echo "Available datasets:"
    for ds in "${DATASETS[@]}"; do
        echo "  $ds"
    done
    exit 0
}

if command -v gh &>/dev/null; then
    TOOL=gh
elif command -v curl &>/dev/null; then
    TOOL=curl
else
    echo "Error: need 'gh' or 'curl' installed." >&2
    exit 1
fi

download_one() {
    local name="$1"
    local tarball="openedge-testdata-${name}.tar.gz"

    echo ">> Downloading ${name} ..."

    if [ "$TOOL" = "gh" ]; then
        if [ "$VERSION" = "latest" ]; then
            gh release download --repo "$REPO" --pattern "$tarball" --clobber --dir /tmp
        else
            gh release download "$VERSION" --repo "$REPO" --pattern "$tarball" --clobber --dir /tmp
        fi
    else
        local url
        if [ "$VERSION" = "latest" ]; then
            url="https://github.com/${REPO}/releases/latest/download/${tarball}"
        else
            url="https://github.com/${REPO}/releases/download/${VERSION}/${tarball}"
        fi
        curl -fSL -o "/tmp/${tarball}" "$url"
    fi

    tar xzf "/tmp/${tarball}" -C "$SCRIPT_DIR"
    rm -f "/tmp/${tarball}"
    echo "   ${name} done."
}

# --- Parse arguments ---------------------------------------------------

targets=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)     list_datasets ;;
        --help|-h)  usage ;;
        --version)  VERSION="$2"; shift 2 ;;
        *)          targets+=("$1"); shift ;;
    esac
done

if [ ${#targets[@]} -eq 0 ]; then
    targets=("${DATASETS[@]}")
fi

# --- Download ----------------------------------------------------------

for t in "${targets[@]}"; do
    found=0
    for ds in "${DATASETS[@]}"; do
        if [ "$t" = "$ds" ]; then found=1; break; fi
    done
    if [ "$found" -eq 0 ]; then
        echo "Error: unknown dataset '$t'. Run '$0 --list' to see options." >&2
        exit 1
    fi
    download_one "$t"
done

echo ""
echo "All requested test data is in place."
