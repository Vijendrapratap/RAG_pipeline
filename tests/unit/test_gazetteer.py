"""Tests for the corpus-mined entity gazetteer (PRD Phase 17, Stage 2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import gazetteer as gz


# --- phonetic skeleton -------------------------------------------------------

def test_singh_variants_share_a_skeleton():
    # ग vs ह final coda after a nasal: both are "Singh".
    assert gz.skeleton("सिंग") == gz.skeleton("सिंह")


def test_puran_variants_share_a_skeleton():
    # retroflex ण vs dental न, and the dropped matra: both are "Puran".
    assert gz.skeleton("पूर्ण") == gz.skeleton("पूरन")


def test_dancing_does_not_key_like_singh():
    # डांसिंग must never collapse onto सिंह — the substring hazard lives only in
    # a naive replace, and the skeleton keeps them distinct too.
    assert gz.skeleton("डांसिंग") != gz.skeleton("सिंह")


def test_phrase_skeleton_matches_across_spellings():
    assert gz.phrase_skeleton(["पूर्ण", "सिंग"]) == gz.phrase_skeleton(["पूरन", "सिंह"])


def test_distinct_names_do_not_share_a_phrase_skeleton():
    assert gz.phrase_skeleton(["राम", "सिंह"]) != gz.phrase_skeleton(["पूरन", "सिंह"])


# --- token classification ----------------------------------------------------

@pytest.mark.parametrize("tok", ["पूरन", "सिंह", "मेडिटेशन", "आया।"])
def test_is_devanagari_word_accepts(tok):
    assert gz.is_devanagari_word(tok)


@pytest.mark.parametrize("tok", ["meditation", "मEDITATION", "2015", "१२", "(Master)"])
def test_is_devanagari_word_rejects_mixed_or_nonword(tok):
    assert not gz.is_devanagari_word(tok)


def test_tokens_keeps_only_devanagari():
    toks = gz._tokens("पूरन सिंह ji said meditation ठीक 2015")
    assert toks == ["पूरन", "सिंह", "ठीक"]


# --- entity candidacy --------------------------------------------------------

def test_tail_anchor_selects_ngram():
    grams = gz._entity_ngrams(["वो", "पूरन", "सिंह", "बोले"])
    assert ("पूरन", "सिंह") in grams


def test_stopword_in_name_slot_is_rejected():
    # `एक सिंह` = "one Singh" — एक is a function word, not a given name.
    assert gz._entity_ngrams(["एक", "सिंह"]) == []


def test_head_honorific_is_not_an_anchor():
    # `गुरु कहा` = "guru said" — a title + verb, never a name. This was the
    # dominant source of noise when head anchoring was tried.
    assert gz._entity_ngrams(["गुरु", "कहा"]) == []


def test_unanchored_bigram_is_not_a_candidate():
    # ordinary prose with no surname
    assert gz._entity_ngrams(["बहुत", "अच्छा"]) == []


# --- apply: boundary safety --------------------------------------------------

def test_apply_rewrites_whole_phrase():
    out, n = gz.apply_gazetteer("वो पूर्ण सिंग जी बोले", {"पूर्ण सिंग": "पूरन सिंह"})
    assert out == "वो पूरन सिंह जी बोले"
    assert n == 1


def test_apply_never_touches_lone_common_word():
    # पूर्ण alone means "complete" and must survive.
    out, n = gz.apply_gazetteer("मेरा काम पूर्ण हुआ", {"पूर्ण सिंग": "पूरन सिंह"})
    assert out == "मेरा काम पूर्ण हुआ"
    assert n == 0


def test_apply_never_fires_inside_a_longer_word():
    # डांसिंग contains सिंग but is one token; a phrase rule cannot reach into it.
    out, n = gz.apply_gazetteer("वो डांसिंग कर रहा", {"अच्छा सिंग": "अच्छा सिंह"})
    assert out == "वो डांसिंग कर रहा"
    assert n == 0


def test_apply_preserves_edge_punctuation():
    out, n = gz.apply_gazetteer("बोले पूर्ण सिंग।", {"पूर्ण सिंग": "पूरन सिंह"})
    assert out == "बोले पूरन सिंह।"
    assert n == 1


def test_apply_prefers_longer_phrase():
    mapping = {"पूर्ण सिंग": "पूरन सिंह", "वारे पूर्ण सिंग": "वाह रे पूरन सिंह"}
    out, _ = gz.apply_gazetteer("वारे पूर्ण सिंग", mapping)
    assert out == "वाह रे पूरन सिंह"


def test_apply_empty_mapping_is_identity():
    assert gz.apply_gazetteer("कुछ भी", {}) == ("कुछ भी", 0)


# --- mine end to end ---------------------------------------------------------

def _write(corpus: Path, name: str, text: str) -> None:
    (corpus / name).write_text(text, encoding="utf-8")


def test_mine_majority_vote(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # canonical पूरन सिंह appears many times across files; पूर्ण सिंग once.
    for i in range(8):
        _write(corpus, f"good_{i}.cleaned.txt", "आज पूरन सिंह जी आए।\n")
    _write(corpus, "bad.cleaned.txt", "वारेपूर्ण के बाद पूर्ण सिंग बोले।\n")

    out = tmp_path / "review.md"
    proposals = gz.mine(corpus, out, gz.MineConfig(min_canonical=5, dominance=4.0))

    assert out.exists()
    pairs = {(p.variant, p.canonical) for p in proposals}
    assert ("पूर्ण सिंग", "पूरन सिंह") in pairs


def test_mine_respects_dominance_gate(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # near-tie: 6 vs 5 does not clear dominance=4.0 -> no proposal.
    for i in range(6):
        _write(corpus, f"a_{i}.cleaned.txt", "पूरन सिंह जी।\n")
    for i in range(5):
        _write(corpus, f"b_{i}.cleaned.txt", "पूर्ण सिंग जी।\n")

    proposals = gz.mine(corpus, tmp_path / "r.md", gz.MineConfig(min_canonical=5, dominance=4.0))
    assert all(not (p.variant == "पूर्ण सिंग") for p in proposals)


def test_mine_requires_minimum_attestation(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # canonical only 3x -> below min_canonical=5 -> whole cluster skipped.
    for i in range(3):
        _write(corpus, f"a_{i}.cleaned.txt", "पूरन सिंह जी।\n")
    _write(corpus, "b.cleaned.txt", "पूर्ण सिंग जी।\n")

    proposals = gz.mine(corpus, tmp_path / "r.md", gz.MineConfig(min_canonical=5, dominance=4.0))
    assert proposals == []


def test_mine_raises_on_empty_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(FileNotFoundError):
        gz.mine(corpus, tmp_path / "r.md")


# --- compile round trip ------------------------------------------------------

def test_compile_parses_surviving_lines(tmp_path: Path):
    review = tmp_path / "review.md"
    review.write_text(
        gz.REVIEW_HEADER
        + 'पूर्ण सिंग → पूरन सिंह   [3 vs 42]   "…पूर्ण सिंग…"\n'
        + '# रद्द किया → कुछ   [1 vs 9]   "struck out"\n'
        + '~~गलत लाइन → गलत~~   [1 vs 9]\n'
        + 'वारेपूर्ण सिंग → वाह रे पूरन सिंह   [1 vs 42]   "…"\n',
        encoding="utf-8",
    )
    mapping = gz.compile_review(review)
    assert mapping == {
        "पूर्ण सिंग": "पूरन सिंह",
        "वारेपूर्ण सिंग": "वाह रे पूरन सिंह",
    }


def test_compile_skips_single_token_variant(tmp_path: Path):
    review = tmp_path / "review.md"
    review.write_text(
        gz.REVIEW_HEADER + 'पूर्ण → पूरन   [3 vs 42]   "…"\n',
        encoding="utf-8",
    )
    assert gz.compile_review(review) == {}


def test_compile_then_write_and_load_round_trips(tmp_path: Path):
    review = tmp_path / "review.md"
    review.write_text(
        gz.REVIEW_HEADER + 'पूर्ण सिंग → पूरन सिंह   [3 vs 42]   "…"\n',
        encoding="utf-8",
    )
    mapping = gz.compile_review(review)
    out = tmp_path / "gazetteer.json"
    gz.write_gazetteer_json(mapping, out)
    assert gz.load_gazetteer(out) == {"पूर्ण सिंग": "पूरन सिंह"}
