# HealthRAG

Sistema multiagente para la recuperación y verificación de evidencia médica sobre
**hipertensión arterial y salud cardiovascular**, basado en RAG híbrido
(BM25 + embeddings + RRF + Cross-Encoder) y dos agentes (Analista y Verificador)
orquestados con LangGraph. Proyecto de PLN III (FIUBA).

> HealthRAG es una herramienta de **consulta documental**. No es un sistema de
> diagnóstico ni da indicaciones para pacientes concretos.

**Informe técnico:** [`docs/Informe_HealthRAG.pdf`](docs/Informe_HealthRAG.pdf).
Incluye objetivo, marco conceptual, corpus, arquitectura, implementación, evaluación,
resultados, conclusiones, consideraciones éticas y referencias.

## Resultados principales

Corpus: 5 guías (Ministerio de Salud de la Nación y OMS), 398 páginas, 1.514 fragmentos.
Evaluación: 29 consultas (24 con evidencia a nivel página y 5 fuera del corpus).

| Recuperación (5 primeros resultados) | Hit | Recall | MRR | Tiempo |
|---|---|---|---|---|
| Vectorial | 0,875 | 0,771 | 0,691 | 49 ms |
| Híbrido BM25 + RRF | 0,917 | 0,854 | 0,743 | 5 ms |
| Híbrido + reranker | 0,958 | 0,875 | 0,727 | 3,7 s |

Con LLM (Groq: `gpt-oss-120b` como analista y `gpt-oss-20b` como verificador), el
ROUGE-L sube de 0,31 a 0,41 respecto del modo extractivo, con fidelidad cercana a 1.
Los tres experimentos con LLM se abstienen bien en las 5 consultas fuera del corpus, y en
EXP3 (multiagente) la decisión final es correcta en las 29 consultas. Detalle y
salvedades en el informe.

## Componentes del proyecto

| Componente | Ubicación |
|---|---|
| API funcional (FastAPI / Streamlit) | `app/api/main.py`, `frontend/streamlit_app.py` |
| Dos agentes con roles y autoevaluación | `app/agents/` (analista, verificador y grafo LangGraph) |
| RAG retriever-ranker con trazabilidad de fuentes | `app/retrieval/` (BM25 + E5 + RRF + cross-encoder); cada cita con documento y página |
| Seguridad (acceso, filtrado de entrada, validación de salida, auditoría) | `app/security/` |
| Flujo modular | `app/` por etapas, `config.yaml`, `notebooks/` 01-04, `tests/` |
| Optimización de costo y latencia | caché TTL, top-k 5, verificador con modelo chico, máximo 2 ciclos (`config.yaml`) |
| Acción final automática | `app/actions/notifier.py` (webhook ante abstenciones) y auditoría JSONL |
| Métricas de modelo, agentes y RAG | `app/evaluation/`, resultados en `reports_*/`, registrados en MLflow (`app/evaluation/mlflow_tracking.py`) |
| Informe técnico | `docs/Informe_HealthRAG.pdf` |

## Estructura

```
HealthRAG/
├── app/
│   ├── config.py              # carga config.yaml + .env
│   ├── ingestion/             # Etapas 1-2: catálogo, extracción, limpieza, chunking
│   ├── retrieval/             # Etapas 3-4: embeddings, FAISS, BM25, RRF, reranker
│   ├── llm/                   # proveedores (OpenAI-compatible / extractivo offline)
│   ├── prompts/prompts.py     # prompts estructurados versionados
│   ├── agents/                # Etapa 5: analista, verificador y grafo LangGraph
│   ├── rag/pipeline.py        # servicio: guardrails -> caché -> agentes -> auditoría
│   ├── security/              # filtros de entrada/salida, API keys, rate limit, auditoría
│   ├── actions/notifier.py    # acción automática opcional (webhook)
│   ├── api/                   # Etapa 6: FastAPI
│   └── evaluation/            # Etapa 7: dataset, métricas, experimentos EXP1-EXP4, registro en MLflow
├── frontend/streamlit_app.py  # demo visual
├── data/
│   ├── raw/                   # los 5 PDF del corpus
│   ├── metadata_template.csv  # metadatos bibliográficos verificados (manual)
│   ├── processed/             # chunks.jsonl (corpus_catalog.csv se genera con make catalog)
│   ├── index/                 # índice FAISS + manifiesto
│   └── evaluation/queries.csv # preguntas de evaluación
├── notebooks/                 # 01-04: usan los módulos, no contienen lógica
├── tests/                     # pytest (corpus sintético, sin internet ni API key)
├── docs/
│   ├── Informe_HealthRAG.pdf  # informe técnico final
│   ├── Informe_HealthRAG.docx # el mismo informe en Word
│   ├── informe/               # generadores del informe (plantilla, scripts, figuras)
│   └── arquitectura.md        # diagrama y descripción de componentes
├── reports_extractivo/        # resultados sin LLM: EXP1-EXP4 (29 consultas)
├── reports_groq/              # resultados con Groq sobre 10 consultas (corrida preliminar)
├── reports_llm_v2/            # resultados con Groq: EXP1 y EXP2 (29 consultas); su EXP3 se descarta (ver NOTA.md)
├── reports_llm_exp3_v2/       # resultados con Groq: EXP3 repetido aparte (29 consultas), los del informe
├── config.yaml  ·  .env.example  ·  requirements.txt  ·  Makefile
```

## Instalación

Con conda (como se desarrolló, Python 3.11):

```bash
conda create -n healthrag python=3.11 -y
conda activate healthrag
pip install -r requirements.txt
cp .env.example .env                                     # completar si se usa un LLM
```

O con venv: `python -m venv .venv`, activarlo (`.venv\Scripts\activate` en Windows)
y seguir con `pip install -r requirements.txt`.

El repositorio incluye los 5 PDF del corpus en `data/raw/` (`GPCN-HTA-version_breve.pdf`,
`GPCN-IAM.pdf`, `GPCN-prev-cardiovascular.pdf`, `Directriz-OMS-tto-farmaco-HTA.pdf` y
`Guia-OMS-prev-enf-cardiovascular.pdf`). Son documentos públicos del Ministerio de Salud de la
Nación y de la OMS, incluidos con fines académicos; título, organismo y URL de cada uno están en
`data/metadata_template.csv`. También se incluyen los fragmentos (`data/processed/chunks.jsonl`)
y el índice FAISS (`data/index/`), así que la API y la evaluación funcionan sin correr las
Etapas 1 a 3. Para regenerarlos desde los PDF, ver **Uso**.

La primera ejecución descarga de Hugging Face los modelos
`intfloat/multilingual-e5-small` (embeddings y tokenizer) y
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (reranker). Si no hay acceso, el
sistema lo registra en el log y usa alternativas offline (tokenizer por regex,
embeddings por hashing, sin reranker) — **solo sirven para pruebas**, no para
los experimentos del informe.

## Uso

```bash
# 1. Corpus: copiar los PDF a data/raw/ y completar data/metadata_template.csv
make catalog          # Etapa 1 -> data/processed/corpus_catalog.csv
make process          # Etapa 2 -> data/processed/chunks.jsonl
make index            # Etapa 3 -> data/index/

# 2. Servir
make api              # http://localhost:8000/docs
make ui               # http://localhost:8501 (en otra terminal)

# 3. Evaluar
make eval             # EXP1-EXP3 -> reports/
make eval-full        # + EXP4 (caché y top-k)
make test
```

Sin `make` (Windows): `python -m app.ingestion.pdf_catalog`,
`python -m app.ingestion.pdf_processor`, `python -m app.retrieval.retriever --rebuild`,
`uvicorn app.api.main:app --reload`, `streamlit run frontend/streamlit_app.py`,
`python -m app.evaluation.run_experiments --exp4`, `python -m pytest`.

### Metadatos del corpus

`data/metadata_template.csv` necesita una fila por PDF. Los `doc_id` deben coincidir
con los de `data/evaluation/queries.csv`:

| doc_id | documento |
|---|---|
| `GPCN_HTA` | Guía de práctica clínica nacional de hipertensión arterial |
| `GPCN_IAM` | Guía de práctica clínica nacional de infarto agudo de miocardio |
| `GPCN_prev_cardiovascular` | Guía de práctica clínica nacional de prevención cardiovascular |
| `Guia_OMS_prev_ECV` | Guía OMS de prevención de enfermedad cardiovascular |
| `Directriz_OMS_HTA` | Directriz OMS de tratamiento farmacológico de la hipertensión |

Título, organización, año, versión y URL se completan con datos verificados de la
fuente; el sistema no los infiere. El CSV ya está completo para los 5 PDF del corpus,
salvo lo marcado en `notes`: año de `GPCN_prev_cardiovascular` (no figura en el PDF) y
URL oficial de `GPCN_HTA`, `Guia_OMS_prev_ECV` y `Directriz_OMS_HTA`.

Resultado del procesamiento (Etapa 2): 1.514 fragmentos de 5 documentos, con
secciones de referencias, autores y similares excluidas (`processing.exclude_sections`).

### Proveedor LLM

| Opción | Configuración en `.env` |
|---|---|
| Sin LLM (extractivo, por defecto) | nada |
| OpenAI | `HEALTHRAG_LLM_PROVIDER=openai_compatible`, `OPENAI_API_KEY=...` |
| Ollama local | además `HEALTHRAG_LLM_BASE_URL=http://localhost:11434/v1`, `HEALTHRAG_ANALYST_MODEL=llama3.1:8b` |
| Groq | `HEALTHRAG_LLM_BASE_URL=https://api.groq.com/openai/v1` y la key de Groq |

El verificador puede usar un modelo más chico que el analista
(`HEALTHRAG_VERIFIER_MODEL`) para reducir costo y latencia.

### API

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "cuanta sal se recomienda consumir por dia", "mode": "multiagent", "include_trace": true}'
```

`mode`: `basic` (EXP1), `advanced` (EXP2) o `multiagent` (EXP3). Otros endpoints:
`GET /health`, `GET /sources/{chunk_id}`, `GET /audit` (rol admin).
Con `security.require_api_key: true` se exige el header `X-API-Key`
(claves en `HEALTHRAG_API_KEYS="clave1,clave2:admin"`).

## Evaluación

`data/evaluation/queries.csv` contiene las 24 consultas de `querys-ejemplos.xlsx` más
5 consultas fuera del corpus (para medir abstención). Columna `relevant`:
`DOC_ID` (cualquier página) o `DOC_ID:PAG` separados por `;`.

Las 24 consultas del Excel tienen páginas de evidencia y respuesta de referencia
verificadas contra los PDF (columna `notes`); conviene que alguien con formación
sanitaria las revise. Como mejora futura queda agregar preguntas de síntesis y
comparación.

Salidas en `reports/` (o en la carpeta de `--out`): `retrieval_summary.csv`,
`e2e_summary.csv`, `e2e_by_category.csv`, `summary.md`, gráficos PNG y
`human_review_sample.csv` (muestra para revisar a mano fidelidad y citas).

```bash
python -m app.evaluation.run_experiments --exp4 --out reports_extractivo   # sin LLM
python -m app.evaluation.run_experiments --out reports_llm_v2              # con LLM
python -m app.evaluation.run_experiments --exps EXP3 --out reports_llm_exp3_v2  # repetir un experimento
python docs/informe/build_informe.py                                       # regenerar el informe (HTML y PDF)
python docs/informe/build_word.py                                          # versión Word (Windows con Microsoft Word)
```

### Seguimiento con MLflow

Las corridas oficiales (`reports_extractivo`, `reports_groq`, `reports_llm_v2` sin su EXP3 y
`reports_llm_exp3_v2`) se registran en MLflow, en el experimento `HealthRAG`. Cada carpeta
es una corrida padre con la configuración como parámetros y los CSV y gráficos como
artefactos. Adentro hay una corrida hija por experimento (EXP1-EXP4) y otra por modo de
recuperación. El comando se puede repetir: lo que ya está registrado no se duplica.

```bash
python -m app.evaluation.mlflow_tracking                     # registrar las corridas oficiales
mlflow ui --backend-store-uri sqlite:///mlflow.db            # interfaz en http://localhost:5000
python -m app.evaluation.run_experiments --out reports_x --mlflow   # registrar una corrida nueva
```

`mlflow.db` y `mlruns/` no se versionan porque guardan rutas absolutas de la PC; se
regeneran con el primer comando.

**Límites de Groq (plan gratuito):** 8.000 tokens por minuto y 200.000 por día por
modelo. Una corrida de EXP1-EXP3 sobre 29 consultas usa cerca de 300.000 tokens entre
los dos modelos, casi todos del analista, así que no entra en un día: conviene correr EXP3 por separado con `--exps`. Si
se agota el cupo, el sistema sigue respondiendo en modo extractivo y lo registra en el
log; antes de usar una corrida, revisar que `tokens > 0` en `e2e_per_query.csv`.

## Seguridad

- Filtro de entrada: longitud, caracteres de control, prompt injection, datos
  personales (DNI, CUIL, email, teléfono, historia clínica) y aviso en casos individuales.
- Los fragmentos recuperados se tratan como datos: se eliminan oraciones con
  instrucciones al modelo antes de pasarlos al LLM.
- Validación de salida: se eliminan citas inexistentes; una respuesta sin citas se
  convierte en abstención.
- Control de acceso por API key con roles, límite de consultas por minuto y
  auditoría JSONL (`logs/audit.jsonl`).
