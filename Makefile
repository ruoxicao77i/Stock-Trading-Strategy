#check data_clean.py load_and_clean() can read scv
#check return column has produced
#check return 
#check add_features() produce Beta_30 Corr_30 Excess_Return

#if the venv shot down you can input:
#source .venv/bin/activate
#python -c "import numpy, pandas, yfinance; print(numpy.__version__, pandas.__version__, yfinance.__version__)"

PYTHON = python3
VENV = .venv
PIP = $(VENV)/bin/pip
PY = $(VENV)/bin/python

.PHONY: install download clean-data run notebooks test clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download:
	$(PY) src/data_download.py

clean-data:
	$(PY) src/data_clean.py

run: install download clean-data notebooks

notebooks:
	$(PIP) install jupyter nbconvert
	$(PY) -m jupyter nbconvert --to notebook --execute src/data_testsearch.ipynb --output data_testsearch_executed.ipynb --output-dir src
	$(PY) -m jupyter nbconvert --to notebook --execute src/data_visualization.ipynb --output data_visualization_executed.ipynb --output-dir src
	$(PY) -m jupyter nbconvert --to notebook --execute src/model.ipynb --output model_executed.ipynb --output-dir src

test:
	$(PIP) install pytest
	$(PY) -m pytest tests

clean:
	rm -rf $(VENV)
	rm -rf data/raw/*
	rm -rf data/clean/*
	rm -f src/*_executed.ipynb