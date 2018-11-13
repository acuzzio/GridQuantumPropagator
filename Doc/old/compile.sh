#!/bin/bash
pdflatex LiegiInitialReport.tex
bibtex LiegiInitialReport.aux
pdflatex LiegiInitialReport.tex
pdflatex LiegiInitialReport.tex

