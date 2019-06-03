OUTPUT_DIR := dist


clean:
	rm -rf allopy.egg-info build dist htmlcov .coverage

test:
	python -m pytest tests/


wheel:
	python setup.py bdist_wheel
