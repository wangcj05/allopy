OUTPUT_DIR := dist

conda:
	conda build --output-folder $(OUTPUT_DIR) conda.recipe


clean:
	rm -rf allopy.egg-info build dist/* htmlcov
	find . -name .coverage -type f -exec rm {} +
	find . -name .ipynb_checkpoints -type d -exec rm -rf {} +


test:
	python -m pytest tests/


wheel:
	python setup.py bdist_wheel
