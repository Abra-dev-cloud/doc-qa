.PHONY: install run test lint clean

install:
	pip install -r requirements.txt

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check app/ tests/
	mypy app/ --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf chroma_db/
