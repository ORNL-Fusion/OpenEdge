# OpenEdge manual, SPARTA style: doc/*.txt -> doc/*.html -> doc/Manual.pdf
#   make html     rebuild every HTML page from its .txt source
#   make pdf      rebuild HTML, then Manual.pdf with htmldoc
#   make clean    remove generated HTML and PDF
# Needs the lammpsdoc txt2html converter in $(PY) and htmldoc on PATH.
PY       ?= python3
TXT2HTML  = $(PY) -m lammpsdoc.txt2html
HTMLDOC  ?= htmldoc

SOURCES  := $(wildcard *.txt)
HTML     := $(SOURCES:.txt=.html)
SECTIONS := Manual Section_intro Section_start Section_commands \
            Section_howto Section_example Section_errors
PAGES    := $(filter-out $(addsuffix .html,$(SECTIONS)),$(HTML))
BOOK     := $(addsuffix .html,$(SECTIONS)) $(sort $(PAGES))

html: $(HTML)

%.html: %.txt
	$(TXT2HTML) -b $< > $@

pdf: html
	$(HTMLDOC) --batch --book --title --toctitle "Table of Contents" \
	  --tocheader .t. --tocfooter ..i --toclevels 4 \
	  --header .t. --footer ..1 --size letter --linkstyle plain \
	  --linkcolor blue --charset utf-8 --compression=9 \
	  -f Manual.pdf $(BOOK)

clean:
	rm -f $(HTML) Manual.pdf

.PHONY: html pdf clean

# Sphinx manual (Read the Docs): doc/src/*.rst -> doc/html_sphinx
# Pages that start with ".. hand-edited" are maintained in doc/src and are
# the source of truth; the remaining (stock SPARTA) pages come from doc/*.txt.
#   make rst      regenerate the non-hand-edited doc/src/*.rst from .txt
#   make sphinx   build the HTML manual with Sphinx into doc/html_sphinx
# Needs `pip install -r doc/utils/requirements.txt`.
SPHINX ?= sphinx-build

rst:
	$(PY) utils/txt2rst.py -o src $(SOURCES)

sphinx: rst
	$(SPHINX) -W --keep-going -b html src html_sphinx

.PHONY: rst sphinx
