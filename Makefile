OUTPUT_DIR := dist

test:
# Assume a 4 core computer to run the tests
	python -m pytest tests/
