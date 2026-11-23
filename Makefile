PY ?= python

.PHONY: install catalog process index data api ui eval eval-full test clean-index

install:            ## Instala dependencias
	$(PY) -m pip install -r requirements.txt

catalog:            ## Etapa 1: catálogo del corpus
	$(PY) -m app.ingestion.pdf_catalog

process:            ## Etapa 2: limpieza y chunking -> data/processed/chunks.jsonl
	$(PY) -m app.ingestion.pdf_processor

index:              ## Etapa 3: embeddings + índice FAISS
	$(PY) -m app.retrieval.retriever --rebuild

data: catalog process index   ## Pipeline documental completo

api:                ## Etapa 6: API en http://localhost:8000/docs
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

ui:                 ## Interfaz Streamlit (requiere la API levantada)
	streamlit run frontend/streamlit_app.py

eval:               ## Etapa 7: EXP1-EXP3
	$(PY) -m app.evaluation.run_experiments

eval-full:          ## EXP1-EXP4 (caché y top-k)
	$(PY) -m app.evaluation.run_experiments --exp4

test:
	$(PY) -m pytest -q

clean-index:
	rm -rf data/index/*
