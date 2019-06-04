OUTPUT_DIR := dist

conda:
	conda build --output-folder $(OUTPUT_DIR) conda.recipe


clean:
	rm -rf allopy.egg-info build dist/* htmlcov .coverage


test:
	python -m pytest tests/


wheel:
	python setup.py bdist_wheel
