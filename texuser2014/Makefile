export
MAINMATTER = texuser-2012-lt
TEMP = temp
PAGE = 1-

LATEX = platex -interaction=batchmode -shell-escape
DVIPDFMX = dvipdfmx
MAP = -f pdftex_ndl14 -f dvipdfm_ndl14 -f hiraginox

TEXINPUTS := latexlib//:
TEXFONTS  := latexlib/fonts//:

all: $(MAINMATTER).dvi
	$(DVIPDFMX) $(MAP) -s $(PAGE) -p "128mm,96mm"  -o $(MAINMATTER).pdf $(MAINMATTER)
	rm -f $(MAINMATTER).dvi

.tex.dvi:
	- $(LATEX) $*
	- $(LATEX) $*

clean:
	rm -f *.aux *.nav *.snm *.log *.toc *.out
