# Resultados de evaluación — HealthRAG

- Consultas: 29 (con evidencia de referencia: 24; a nivel página: 24)
- Proveedor LLM: openai_compatible · Embeddings: intfloat/multilingual-e5-small · Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Duración total: 850.1 s

## Recuperación

| retrieval_mode   |   retrieval_ms |   useful_context@5 |    mrr |   hit@5 |   recall@5 |   precision@5 |   ndcg@5 |   hit@10 |   recall@10 |   precision@10 |   ndcg@10 |
|:-----------------|---------------:|-------------------:|-------:|--------:|-----------:|--------------:|---------:|---------:|------------:|---------------:|----------:|
| advanced         |      3427.1    |             0.3333 | 0.7267 |  0.9583 |     0.875  |        0.3333 |   0.6246 |   1      |      0.9583 |         0.225  |    0.7199 |
| basic            |        58.1497 |             0.3083 | 0.691  |  0.875  |     0.7708 |        0.3083 |   0.5901 |   0.9583 |      0.9375 |         0.2333 |    0.7131 |
| hybrid           |         4.1594 |             0.35   | 0.7431 |  0.9167 |     0.8542 |        0.35   |   0.659  |   0.9583 |      0.9375 |         0.2375 |    0.7554 |

(tabla completa con k=1,3,5,10 en `retrieval_summary.csv`)

## Respuestas (EXP1–EXP4)

| exp   | mode       |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |   bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:-----------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|-------:|---------------:|-----------------:|-----------------:|
| EXP3  | multiagent |  29 |      0.1724 |                    1 |                  0 |         0.9948 |                0.902 |                   0.6264 |        0.9583 |       0.0086 |       1.2759 |  5028.97 |     0.0013 |   0.3816 | 0.1778 |         0.9357 |          16634.2 |          67136.9 |

## Por categoría de pregunta

| exp   | category      |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |     bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:--------------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|---------:|---------------:|-----------------:|-----------------:|
| EXP3  | directa       |  24 |           0 |                  nan |                  0 |         0.9948 |                0.902 |                   0.6264 |        0.9583 |       0.0104 |        1.125 |  5082.29 |     0.0014 |   0.3816 |   0.1778 |         0.9357 |          15304   |          74628.9 |
| EXP3  | sin_evidencia |   5 |           1 |                    1 |                nan |       nan      |              nan     |                 nan      |      nan      |       0      |        2     |  4773    |     0.0009 | nan      | nan      |       nan      |          33530.4 |          37457   |

> Fidelidad y precisión de citas se estiman con un verificador léxico-numérico independiente del LLM; complementar con la revisión humana de `human_review_sample.csv`. Si las consultas tienen relevancia solo a nivel documento, recall/precision son optimistas.