import pytest


@pytest.fixture(autouse=True)
def media_root(tmp_path, settings):
    settings.MEDIA_ROOT = tmp_path / "media"
    return settings.MEDIA_ROOT


@pytest.fixture(autouse=True)
def fake_eval_query_embedder(monkeypatch, fake_query_embedder):
    """The eval embeds all questions up front, outside documents.retrieval."""
    monkeypatch.setattr("evals.retrieval_eval.get_embedder", lambda: fake_query_embedder)
    return fake_query_embedder
