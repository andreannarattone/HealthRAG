from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.retriever import HealthRetriever


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert ids[0] == "b"                       # aparece alto en ambas listas
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_weights_validation():
    import pytest
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_bm25_finds_exact_terms(retriever):
    hits = retriever.bm25_search("electrocardiograma dolor torácico", 3)
    assert retriever.chunks[hits[0][0]].doc_id == "TEST_IAM"


def test_modes_return_traceable_results(retriever):
    for mode in ("basic", "hybrid", "advanced"):
        res = retriever.retrieve("¿cuánta sal por día?", mode=mode, top_k=3)
        assert 1 <= len(res.chunks) <= 3
        assert res.chunks[0].rank == 1
        assert "total_ms" in res.timings_ms
        d = res.chunks[0].to_dict()
        assert {"chunk_id", "doc_id", "page", "citation"} <= d.keys()
    hybrid = retriever.retrieve("consumo de sal 5 gramos", mode="hybrid", top_k=1)
    assert "5 gramos" in hybrid.chunks[0].chunk.text
    assert "rrf" in hybrid.chunks[0].scores


def test_exclude_ids(retriever):
    first = retriever.retrieve("sal", mode="hybrid", top_k=2)
    second = retriever.retrieve("sal", mode="hybrid", top_k=2, exclude_ids=set(first.chunk_ids))
    assert not set(first.chunk_ids) & set(second.chunk_ids)


def test_index_is_persisted_and_reused(cfg, retriever):
    again = HealthRetriever(cfg)
    assert again.vector_index.size == len(retriever.chunks)


def test_reranker_is_used_when_available(cfg):
    class ReverseReranker:
        name = "fake"

        def score(self, query, passages):
            return [float(-i) for i in range(len(passages))][::-1]  # invierte el orden

    r = HealthRetriever(cfg, reranker=ReverseReranker())
    hybrid = r.retrieve("sal", mode="hybrid", top_k=20)
    advanced = r.retrieve("sal", mode="advanced", top_k=20)
    assert advanced.chunk_ids[0] == hybrid.chunk_ids[-1]
    assert "rerank" in advanced.chunks[0].scores
