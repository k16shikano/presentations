export
MAINMATTER = void
TEMP = temp
PAGE = 1-

LATEX = uplatex -interaction=batchmode -shell-escape
DVIPDFMX = dvipdfmx
MAP = -f presenbold -f lucida

TEXINPUTS := latexlib//:
TEXFONTS  := fonts//:

all: $(MAINMATTER).dvi
	$(DVIPDFMX) $(MAP) -s $(PAGE) -p "128mm,96mm"  -o $(MAINMATTER).pdf $(MAINMATTER)
	rm -f $(MAINMATTER).dvi

.tex.dvi:
	- $(LATEX) $*
	- $(LATEX) $*
#	bibtex $*
#	- $(LATEX) $*
#	- $(LATEX) $*

clean:
	rm -f *.aux *.nav *.snm *.log *.toc *.out *.xbb
