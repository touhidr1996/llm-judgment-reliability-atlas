.PHONY: setup run test api
setup:
	python -m pip install -e ".[dev]"
run:
	PYTHONPATH=src python -m judgment_atlas.pipeline
test:
	PYTHONPATH=src pytest -q
api:
	uvicorn judgment_atlas.api:app --app-dir src --reload
