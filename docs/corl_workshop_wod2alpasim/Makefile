TEX := paper

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(TEX).bib ../corl2027/corl_2026.sty ../corl2027/corlabbrvnat.bst
	pdflatex $(TEX)
	BSTINPUTS=../corl2027: bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf
