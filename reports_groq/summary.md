# Resultados de evaluación — HealthRAG

- Consultas: 10 (con evidencia de referencia: 7; a nivel página: 7)
- Proveedor LLM: openai_compatible · Embeddings: intfloat/multilingual-e5-small · Reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- Duración total: 588.5 s

## Recuperación

| retrieval_mode   |   retrieval_ms |   useful_context@5 |    mrr |   hit@5 |   recall@5 |   precision@5 |   ndcg@5 |   hit@10 |   recall@10 |   precision@10 |   ndcg@10 |
|:-----------------|---------------:|-------------------:|-------:|--------:|-----------:|--------------:|---------:|---------:|------------:|---------------:|----------:|
| advanced         |      3930.63   |             0.3429 | 0.9048 |       1 |     0.8571 |        0.3429 |   0.6407 |        1 |           1 |         0.3    |    0.838  |
| basic            |        90.4949 |             0.4857 | 0.8333 |       1 |     0.9286 |        0.4857 |   0.7776 |        1 |           1 |         0.3143 |    0.8623 |
| hybrid           |         5.3896 |             0.5429 | 1      |       1 |     1      |        0.5429 |   0.9073 |        1 |           1 |         0.3286 |    0.9684 |

(tabla completa con k=1,3,5,10 en `retrieval_summary.csv`)

## Respuestas (EXP1–EXP4)

| exp   | mode       |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |   bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:-----------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|-------:|---------------:|-----------------:|-----------------:|
| EXP1  | basic      |  10 |         0.3 |                    1 |             0      |         0.9643 |               0.8214 |                   0.8214 |             1 |       0.0333 |          1   |   2081   |     0.0005 |   0.5543 | 0.2639 |         0.948  |          9135.32 |          25941   |
| EXP2  | advanced   |  10 |         0.4 |                    1 |             0.1429 |         1      |               0.8611 |                   0.5833 |             1 |       0.025  |          1   |   2076.8 |     0.0005 |   0.4982 | 0.2524 |         0.9427 |         13017.5  |          27739.1 |
| EXP3  | multiagent |  10 |         0.3 |                    1 |             0      |         1      |               0.8929 |                   0.5238 |             1 |       0.05   |          1.4 |   4704.2 |     0.0011 |   0.5302 | 0.2425 |         0.944  |         16794.8  |          83365.9 |

## Por categoría de pregunta

| exp   | category      |   n |   abstained |   correct_abstention |   false_abstention |   faithfulness |   citation_precision |   cited_source_relevance |   context_hit |   redundancy |   iterations |   tokens |   cost_usd |   rougeL |     bleu |   semantic_sim |   latency_p50_ms |   latency_p95_ms |
|:------|:--------------|----:|------------:|---------------------:|-------------------:|---------------:|---------------------:|-------------------------:|--------------:|-------------:|-------------:|---------:|-----------:|---------:|---------:|---------------:|-----------------:|-----------------:|
| EXP1  | directa       |   7 |      0      |                  nan |             0      |         0.9643 |               0.8214 |                   0.8214 |             1 |       0.0476 |       1      |  2207.43 |     0.0006 |   0.5543 |   0.2639 |         0.948  |          5417.25 |          28007.3 |
| EXP1  | sin_evidencia |   3 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |           nan |       0      |       1      |  1786    |     0.0004 | nan      | nan      |       nan      |         14217.6  |          14827.3 |
| EXP2  | directa       |   7 |      0.1429 |                  nan |             0.1429 |         1      |               0.8611 |                   0.5833 |             1 |       0.0357 |       1      |  2081.43 |     0.0005 |   0.4982 |   0.2524 |         0.9427 |         13138.8  |          28496.2 |
| EXP2  | sin_evidencia |   3 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |           nan |       0      |       1      |  2066    |     0.0005 | nan      | nan      |       nan      |         12532    |          14517.4 |
| EXP3  | directa       |   7 |      0      |                  nan |             0      |         1      |               0.8929 |                   0.5238 |             1 |       0.0714 |       1.1429 |  4617.14 |     0.0012 |   0.5302 |   0.2425 |         0.944  |         14703.9  |          91223.1 |
| EXP3  | sin_evidencia |   3 |      1      |                    1 |           nan      |       nan      |             nan      |                 nan      |           nan |       0      |       2      |  4907.33 |     0.001  | nan      | nan      |       nan      |         33166.4  |          35411.2 |

> Fidelidad y precisión de citas se estiman con un verificador léxico-numérico independiente del LLM; complementar con la revisión humana de `human_review_sample.csv`. Si las consultas tienen relevancia solo a nivel documento, recall/precision son optimistas.