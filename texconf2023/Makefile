export
MAINMATTER = presentation
TEMP = temp
PAGE = 1-

LATEX = lualatex -interaction=batchmode -shell-escape
DVIPDFMX = dvipdfmx
MAP = -f presenbold -f otf-up-hiragino -f lucida

TEXINPUTS := latexlib//:
TEXFONTS  := fonts//:

all: $(MAINMATTER).dvi
#	$(DVIPDFMX) $(MAP) -s $(PAGE) -p "128mm,96mm"  -o $(MAINMATTER).pdf $(MAINMATTER)
	rm -f $(MAINMATTER).dvi

.tex.dvi:
	- $(LATEX) $*
	- $(LATEX) $*

clean:
	rm -f *.aux *.nav *.snm *.log *.toc *.out *.xbb
