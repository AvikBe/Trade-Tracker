.PHONY: install db migrate test

install:
	python3 -m pip install -e '.[dev]'

db:
	docker compose up -d db

migrate:
	tt migrate

test:
	pytest -q
