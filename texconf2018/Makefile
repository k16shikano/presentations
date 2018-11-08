IN=image.eps

all:
#	purifyeps --fontmap=mpost.fmp $(IN) temp.mps
#	java -jar eps2pgf.jar temp.mps -o image.tex
	uplatex body.tex
	dvipdfmx -f otf-up-hiragino -f lucida -o body.pdf body.dvi



