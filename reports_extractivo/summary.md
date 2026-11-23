# Resultados de evaluación — HealthRAG

- Consultas: 29 (con evidencia de referencia: 24; a nivel página: 24)
- Proveedor LLM: extractive · Embeddings: intfloat/multilingual-e5-small · Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Duración total: 854.4 s

## Recuperación

| retrieval_mode   |   retrieval_ms |   useful_context@5 |    mrr |   hit@5 |   recall@5 |   precision@5 |   ndcg@5 |   hit@10 |   recall@10 |   precision@10 |   ndcg@10 |
|:-----------------|---------------:|-------------------:|-------:|--------:|-----------:|--------------:|---------:|---------:|------------:|---------------:|----------:|
| advanced         |      3693.85   |             0.3333 | 0.7267 |  0.9583 |     0.875  |        0.3333 |   0.6246 |   1      |      0.9583 |         0.225  |    0.7199 |
| basic            |        49.3485 |             0.3083 | 0.691  |  0.875  |     0.7708 |        0.3083 |   0.5901 |   0.9583 |      0.9375 |         0.2333 |    0.7131 |
| hybrid           |         4.5985 |             0.35   | 0.7431 |  0.9167 |     0.8542 |        0.35   |   0.659  |   0.9583 |      0.9375 |         0.2375 |    0.7554 |

(tabla completa con k=1,3,5,10 en `retrieval_summary.csv`)

## Respuestas (EXP1–EXP4)

| exp                 | mode       |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |   bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:--------------------|:-----------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|-------:|---------------:|-----------------:|-----------------:|
| EXP1                | basic      |  29 |      0.2414 |                  1   |             0.0833 |              1 |                    1 |                   0.5909 |        0.875  |       0      |       1      |        0 |          0 |   0.3054 | 0.1362 |         0.9203 |          10.736  |          65.7645 |
| EXP2                | advanced   |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0287 |       1      |        0 |          0 |   0.2876 | 0.1156 |         0.9223 |        3757.04   |        5494.69   |
| EXP3                | multiagent |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0287 |       1.2069 |        0 |          0 |   0.2876 | 0.1156 |         0.9223 |        3798.89   |        9751.17   |
| EXP4_cache_caliente | multiagent |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0287 |       1.2069 |        0 |          0 |   0.2876 | 0.1156 |         0.9223 |           0.2687 |           0.4445 |
| EXP4_cache_frio     | multiagent |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0287 |       1.2069 |        0 |          0 |   0.2876 | 0.1156 |         0.9223 |        4383.92   |        9100.91   |
| EXP4_top3           | multiagent |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.6136 |        0.9167 |       0.0287 |       1.2759 |        0 |          0 |   0.2907 | 0.1189 |         0.9222 |        3956.42   |        8363.79   |
| EXP4_top8           | multiagent |  29 |      0.2069 |                  0.8 |             0.0833 |              1 |                    1 |                   0.5379 |        1      |       0.0287 |       1.2069 |        0 |          0 |   0.2621 | 0.1003 |         0.9196 |        4473.9    |        8588.05   |

## Por categoría de pregunta

| exp   | category      |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |     bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:--------------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|---------:|---------------:|-----------------:|-----------------:|
| EXP1  | directa       |  24 |      0.0833 |                nan   |             0.0833 |              1 |                    1 |                   0.5909 |        0.875  |       0      |       1      |        0 |          0 |   0.3054 |   0.1362 |         0.9203 |          10.2679 |          11.6737 |
| EXP1  | sin_evidencia |   5 |      1      |                  1   |           nan      |            nan |                  nan |                 nan      |      nan      |       0      |       1      |        0 |          0 | nan      | nan      |       nan      |          63.1393 |          67.7193 |
| EXP2  | directa       |  24 |      0.0833 |                nan   |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0347 |       1      |        0 |          0 |   0.2876 |   0.1156 |         0.9223 |        3742.4    |        5550.47   |
| EXP2  | sin_evidencia |   5 |      0.8    |                  0.8 |           nan      |              1 |                    1 |                 nan      |      nan      |       0      |       1      |        0 |          0 | nan      | nan      |       nan      |        3813.83   |        4396.98   |
| EXP3  | directa       |  24 |      0.0833 |                nan   |             0.0833 |              1 |                    1 |                   0.6061 |        0.9583 |       0.0347 |       1.0833 |        0 |          0 |   0.2876 |   0.1156 |         0.9223 |        3786.92   |        6559.02   |
| EXP3  | sin_evidencia |   5 |      0.8    |                  0.8 |           nan      |              1 |                    1 |                 nan      |      nan      |       0      |       1.8    |        0 |          0 | nan      | nan      |       nan      |        9152.15   |       10693.1    |

> Fidelidad y precisión de citas se estiman con un verificador léxico-numérico independiente del LLM; complementar con la revisión humana de `human_review_sample.csv`. Si las consultas tienen relevancia solo a nivel documento, recall/precision son optimistas.