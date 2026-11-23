# Resultados de evaluación — HealthRAG

- Consultas: 29 (con evidencia de referencia: 24; a nivel página: 24)
- Proveedor LLM: openai_compatible · Embeddings: intfloat/multilingual-e5-small · Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Duración total: 1780.3 s

## Recuperación

| retrieval_mode   |   retrieval_ms |   useful_context@5 |    mrr |   hit@5 |   recall@5 |   precision@5 |   ndcg@5 |   hit@10 |   recall@10 |   precision@10 |   ndcg@10 |
|:-----------------|---------------:|-------------------:|-------:|--------:|-----------:|--------------:|---------:|---------:|------------:|---------------:|----------:|
| advanced         |      3859.18   |             0.3333 | 0.7267 |  0.9583 |     0.875  |        0.3333 |   0.6246 |   1      |      0.9583 |         0.225  |    0.7199 |
| basic            |        66.9831 |             0.3083 | 0.691  |  0.875  |     0.7708 |        0.3083 |   0.5901 |   0.9583 |      0.9375 |         0.2333 |    0.7131 |
| hybrid           |         4.5723 |             0.35   | 0.7431 |  0.9167 |     0.8542 |        0.35   |   0.659  |   0.9583 |      0.9375 |         0.2375 |    0.7554 |

(tabla completa con k=1,3,5,10 en `retrieval_summary.csv`)

## Respuestas (EXP1–EXP4)

| exp   | mode       |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |   bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:-----------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|-------:|---------------:|-----------------:|-----------------:|
| EXP1  | basic      |  29 |      0.1724 |                    1 |             0      |         0.9948 |               0.9365 |                   0.6229 |        0.875  |            0 |       1      |  2072.1  |     0.0006 |   0.4107 | 0.1937 |         0.9329 |          14238.1 |          45978.3 |
| EXP2  | advanced   |  29 |      0.2069 |                    1 |             0.0417 |         1      |               0.9238 |                   0.6297 |        0.9583 |            0 |       1      |  2061.79 |     0.0006 |   0.3968 | 0.193  |         0.9358 |          16006.3 |          30524   |
| EXP3  | multiagent |  29 |      0.1724 |                    1 |             0      |         0.9958 |               0.8666 |                   0.6042 |        0.9583 |            0 |       1.3448 |  4088.07 |     0.0011 |   0.3667 | 0.16   |         0.9308 |          13993.9 |          59404.1 |

## Por categoría de pregunta

| exp   | category      |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |     bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:--------------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|---------:|---------------:|-----------------:|-----------------:|
| EXP1  | directa       |  24 |      0      |                  nan |             0      |         0.9948 |               0.9365 |                   0.6229 |        0.875  |            0 |       1      |  2129.79 |     0.0006 |   0.4107 |   0.1937 |         0.9329 |         14465.9  |          49011.6 |
| EXP1  | sin_evidencia |   5 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |      nan      |            0 |       1      |  1795.2  |     0.0004 | nan      | nan      |       nan      |         14007.4  |          14535.5 |
| EXP2  | directa       |  24 |      0.0417 |                  nan |             0.0417 |         1      |               0.9238 |                   0.6297 |        0.9583 |            0 |       1      |  2114.38 |     0.0006 |   0.3968 |   0.193  |         0.9358 |         16889.1  |          31855.3 |
| EXP2  | sin_evidencia |   5 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |      nan      |            0 |       1      |  1809.4  |     0.0004 | nan      | nan      |       nan      |         13623.2  |          17192.1 |
| EXP3  | directa       |  24 |      0      |                  nan |             0      |         0.9958 |               0.8666 |                   0.6042 |        0.9583 |            0 |       1.2083 |  4839    |     0.0013 |   0.3667 |   0.16   |         0.9308 |         16659.8  |          64893.8 |
| EXP3  | sin_evidencia |   5 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |      nan      |            0 |       2      |   483.6  |     0.0001 | nan      | nan      |       nan      |          7595.34 |          25794.9 |

> Fidelidad y precisión de citas se estiman con un verificador léxico-numérico independiente del LLM; complementar con la revisión humana de `human_review_sample.csv`. Si las consultas tienen relevancia solo a nivel documento, recall/precision son optimistas.