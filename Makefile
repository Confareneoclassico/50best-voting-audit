PY ?= python3

.PHONY: all panel part1 bounds part2 part3 extras figures report test archive clean
# the steps of 'all' read each other's results: run them in order even with -j
.NOTPARALLEL:

all: panel part1 bounds part2 part3 extras figures report

panel:
	@mkdir -p results data/derived report/figures
	$(PY) src/lists.py

part1: panel
	$(PY) src/churn.py > results/part1_log.txt

bounds: panel
	$(PY) src/bounds.py > results/bounds_log.txt

part2: part1
	$(PY) src/noise.py > results/part2_log.txt

part3: part2
	$(PY) src/gsa.py > results/part3_log.txt

extras: part3
	$(PY) src/extras.py > results/extras_log.txt

# figures and report use the results already in results/ (run 'make all' first)
figures:
	$(PY) src/figures.py

report: figures
	cd report && for f in audit supplement; do pdflatex -interaction=nonstopmode $$f.tex >/dev/null; done; \
	for f in audit supplement; do bibtex $$f >/dev/null; done; \
	for i in 1 2; do for f in audit supplement; do pdflatex -interaction=nonstopmode $$f.tex >/dev/null; done; done

test:
	$(PY) -m pytest -q tests

# needs network access to web.archive.org: run locally, then make report
archive:
	$(PY) tools/archive_sources.py

clean:
	rm -f results/*.csv results/*.json results/*.npy results/*.npz results/*_log.txt data/derived/panel.csv
	rm -f report/*.aux report/*.bbl report/*.blg report/*.log report/*.out report/audit.pdf report/supplement.pdf
	rm -f report/table_bounds.tex report/table_intervals.tex report/table_ot.tex report/table_effects.tex report/figures/*.pdf
