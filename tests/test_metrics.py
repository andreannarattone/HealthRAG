import pytest

from app.evaluation.dataset import _legacy_label, parse_relevant
from app.evaluation.metrics import bleu, redundancy, retrieval_metrics, rouge_l


def test_retrieval_metrics_page_level():
    retrieved = [("A", 1), ("B", 2), ("A", 5), ("C", 1)]
    m = retrieval_metrics(retrieved, [("A", 5), ("C", 1)], ks=[1, 3, 5])
    assert m["mrr"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert m["precision@3"] == pytest.approx(1 / 3)
    assert 0 < m["ndcg@5"] < 1


def test_retrieval_metrics_doc_level():
    m = retrieval_metrics([("A", 1), ("A", 9)], [("A", None)], ks=[1])
    assert m["hit@1"] == 1 and m["recall@1"] == 1 and m["ndcg@1"] == 1


def test_text_similarity_metrics():
    ref = "Se recomienda consumir menos de 5 gramos de sal por día"
    assert rouge_l(ref, ref) == pytest.approx(1.0)
    assert bleu(ref, ref) == pytest.approx(1.0)
    assert rouge_l("la vacuna antigripal", ref) == 0.0
    assert bleu("algo distinto", ref) < 0.2


def test_redundancy():
    claims = [{"text": "La sal debe reducirse a 5 g"}, {"text": "La sal debe reducirse a 5 g."},
              {"text": "Actividad física 150 minutos"}]
    assert redundancy(claims) == pytest.approx(1 / 3)


def test_dataset_parsing():
    assert parse_relevant("GPCN_IAM:21; GPCN_HTA") == [("GPCN_IAM", 21), ("GPCN_HTA", None)]
    assert _legacy_label("GPCN_IAM_p21") == ("GPCN_IAM", 21)
    assert _legacy_label("GPCN_prev_cardiovascular_p..") == ("GPCN_prev_cardiovascular", None)
