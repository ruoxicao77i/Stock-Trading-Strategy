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
NBCONVERT = $(VENV)/bin/jupyter-nbconvert

.PHONY: install download clean-data prepare run notebooks test clean

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PY) -m ipykernel install --sys-prefix --name cs506 --display-name "Python 3 (CS506)"
	touch $@

install: $(VENV)/.installed

download: $(VENV)/.installed
	$(PY) src/data_download.py

clean-data: $(VENV)/.installed
	$(PY) src/data_clean.py

prepare: $(VENV)/.installed
	$(PY) src/prepare_data.py

run: install download clean-data prepare notebooks

notebooks: $(VENV)/.installed
	$(NBCONVERT) --to notebook --execute --ExecutePreprocessor.kernel_name=cs506 src/data_testsearch.ipynb --output data_testsearch_executed.ipynb --output-dir src
	$(NBCONVERT) --to notebook --execute --ExecutePreprocessor.kernel_name=cs506 src/data_visualization.ipynb --output data_visualization_executed.ipynb --output-dir src
	$(NBCONVERT) --to notebook --execute --ExecutePreprocessor.kernel_name=cs506 src/model.ipynb --output model_executed.ipynb --output-dir src

test:
	$(PIP) install pytest
	$(PY) -m pytest tests

clean:
	rm -rf $(VENV)
	rm -rf data/raw/*
	rm -rf data/clean/*
	rm -f src/*_executed.ipynb