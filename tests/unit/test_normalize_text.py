"""Unit tests for ingestion.normalize_text (PRD Phase 17).

Every rule family has a positive test AND a false-positive test. The
false-positive tests are the point: each anti-rule below names a real surface
form, with its measured corpus frequency, that an obvious-looking cleanup would
have destroyed. They are acceptance criteria, not nice-to-haves.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import normalize_text as nt

RULES_JSON = Path(__file__).parent.parent.parent / "data" / "normalize_rules.json"


@pytest.fixture
def rules() -> nt.Rules:
    return nt.Rules(
        loanword_tokens={"मEDITATION": "मेडिटेशन", "सittings": "सिटिंग्स"},
        native_tokens={"रishi": "रिशी", "Kumarों": "कुमारों"},
        spelling_scatter={"मीडिटेशन": "मेडिटेशन"},
        hallucination_tokens=("सब्सक्राइब", "सबस्क्राइब"),
        hallucination_tail=("करते", "करें", "करो", "कर", "दो", "हैं", "है"),
    )


# ---- rule 1/2/3: exact whole-token maps ---------------------------------


def test_intra_word_script_mix_is_repaired(rules):
    out, n = nt.map_tokens("आज मEDITATION में रishi बैठे", {**rules.loanword_tokens,
                                                            **rules.native_tokens})
    assert out == "आज मेडिटेशन में रिशी बैठे"
    assert n == 2


def test_map_preserves_punctuation_around_the_token(rules):
    """A glued danda must not hide `मEDITATION।` from the map — but the danda
    itself stays. Tantivy already tokenizes across it."""
    out, n = nt.map_tokens("मEDITATION। और मEDITATION,", rules.loanword_tokens)
    assert out == "मेडिटेशन। और मेडिटेशन,"
    assert n == 2


def test_map_is_whole_token_never_substring():
    """डांसिंग (dancing) contains सिंग; डिस्लाइक contains लाइक; पोटेंशियल contains
    टेंशन. A substring pass corrupts all three."""
    table = {"सिंग": "सिंह", "लाइक": "X", "टेंशन": "Y"}
    text = "डांसिंग डिस्लाइक पोटेंशियल"
    out, n = nt.map_tokens(text, table)
    assert out == text
    assert n == 0
    # ...but the bare tokens still map.
    out2, n2 = nt.map_tokens("सिंग", table)
    assert out2 == "सिंह" and n2 == 1


def test_devanagari_spelling_scatter_collapses(rules):
    out, n = nt.map_tokens("मीडिटेशन करो", rules.spelling_scatter)
    assert out == "मेडिटेशन करो"
    assert n == 1


def test_is_script_mixed():
    assert nt.is_script_mixed("मEDITATION")
    assert nt.is_script_mixed("Kumarों")
    assert not nt.is_script_mixed("मेडिटेशन")
    assert not nt.is_script_mixed("meditation")
    assert not nt.is_script_mixed("meditation।")  # punctuation is not a script


# ---- gloss stripping ----------------------------------------------------


def test_latin_glosses_are_stripped_and_captured():
    """4,010 invented parentheticals across 305 files. The speaker never said them."""
    out, glosses = nt.strip_latin_glosses("गुरु (Master) ने तीसरी आँख (Third Eye) कहा")
    assert out == "गुरु ने तीसरी आँख कहा"
    assert glosses == ["Master", "Third Eye"]


def test_devanagari_parenthetical_survives():
    """Only a Latin-alphabet gloss is the cleaner's invention. A Hindi aside is
    the speaker's own."""
    text = "अंतर्मुखी (यानि भीतर की ओर) हो जाओ"
    out, glosses = nt.strip_latin_glosses(text)
    assert out == text
    assert glosses == []


# ---- hallucination excision ---------------------------------------------


def test_subscribe_family_is_excised(rules):
    out, n = nt.excise_hallucination("इस चैनल को सब्सक्राइब करते हैं", rules)
    assert "सब्सक्राइब" not in out
    assert out == "इस चैनल को"
    assert n == 1


def test_subscribe_excision_does_not_eat_following_genuine_text(rules):
    """159 segments glue the artifact to real meditation vocabulary. The match
    must stop at the first word that is not a listed inflection."""
    out, _ = nt.excise_hallucination("सब्सक्राइब देखते रहना भीतर", rules)
    assert out == "देखते रहना भीतर"


def test_subscribe_excision_never_crosses_aur(rules):
    """`और` is deliberately absent from the tail list."""
    out, _ = nt.excise_hallucination("सब्सक्राइब करते हैं और गहरा उतरो", rules)
    assert out == "और गहरा उतरो"


def test_variant_spelling_is_excised(rules):
    """9 cleaned files retain सबस्क्राइब (स-ब-स); both spellings must be listed."""
    out, n = nt.excise_hallucination("सबस्क्राइब करें", rules)
    assert out == "" and n == 1


def test_kar_do_alone_is_not_a_hallucination(rules):
    """`कर दो` (do it) is ordinary Hindi. Only its binding to the subscribe token
    makes it filler."""
    text = "अब तुम कर दो"
    out, n = nt.excise_hallucination(text, rules)
    assert out == text and n == 0


# ---- loop collapse ------------------------------------------------------


def test_long_single_token_run_collapses_to_three():
    out, n = nt.collapse_loops("भीतर " + "झाल " * 10 + "चलो")
    assert out == "भीतर झाल झाल झाल चलो"
    assert n == 1


def test_comma_joined_run_is_the_same_signature():
    """The cleaner emits both `ओ ओ ओ` and `ओ, ओ, ओ,`. One signature."""
    out, n = nt.collapse_loops("ओ, " * 8)
    assert out == "ओ, ओ, ओ,"
    assert n == 1


def test_three_om_survives_loop_collapse():
    """A chanted `ओम् ओम् ओम्` is not an ASR artifact. 3,648 length-3 runs in the
    corpus are legitimate."""
    for text in ("ओम् ओम् ओम्", "बहुत बहुत बहुत", "नमो नमो नमो", "ॐ ॐ ॐ"):
        out, n = nt.collapse_loops(text)
        assert out == text, text
        assert n == 0


def test_five_repeats_survive_but_six_collapse():
    """N=6 is the measured boundary, not a round number."""
    assert nt.collapse_loops("राम " * 5)[1] == 0
    assert nt.collapse_loops("राम " * 6)[1] == 1


def test_loop_collapse_refuses_unsafe_thresholds():
    """71,240 length-2 runs are legitimate reduplication. A caller asking to
    collapse them has misunderstood the rule; do not silently clamp."""
    with pytest.raises(ValueError, match="reduplication"):
        nt.collapse_loops("राम राम", min_run=2)
    with pytest.raises(ValueError, match="reduplication"):
        nt.collapse_loops("राम " * 8, keep=1)


def test_non_adjacent_repeats_are_untouched():
    """Only a *consecutive* run is a loop. A phrase loop (`ब्लूम लाइक` xN) has no
    token adjacent to itself and is a documented coverage gap."""
    text = "ब्लूम लाइक " * 8
    out, n = nt.collapse_loops(text)
    assert out.split() == text.split()
    assert n == 0


# ---- end-to-end + rule ordering -----------------------------------------


def test_subscribe_loop_is_deleted_not_collapsed_to_three(rules):
    """Excision runs before collapse. Otherwise three copies of the artifact
    survive into the index."""
    res = nt.normalize("सब्सक्राइब " * 9 + "भीतर", rules)
    assert "सब्सक्राइब" not in res.text
    assert res.text == "भीतर"


def test_normalize_reports_what_it_changed(rules):
    res = nt.normalize("मEDITATION (Master) सब्सक्राइब करो " + "झाल " * 7, rules)
    assert res.counts["loanword_tokens"] == 1
    assert res.counts["glosses"] == 1
    assert res.counts["hallucination"] == 1
    assert res.counts["loops"] == 1
    assert res.glosses == ["Master"]
    assert "मेडिटेशन" in res.text and "Master" not in res.text


# ---- ANTI-RULES: measured surface forms that must survive untouched ------
#
# Each line is a rule someone will eventually propose. Each would destroy real
# content. The corpus frequencies are from the full 7,413 `.cleaned.txt`.


@pytest.mark.parametrize("text,why", [
    ("विचारों को देखते रहना", "7,928 occ / 1,413 files — THE core meditation instruction"),
    ("चौबीस घंटे बैठे रहे", "घंटे = hours (2,128 occ), not the YouTube bell"),
    ("जीने लाइक हो जाओगे", "लाइक = -worthy (557 legit occ), not a YouTube like"),
    ("आस्था चैनल पर आता था", "चैनल = a real TV channel (65 occ), not 'subscribe to my channel'"),
    ("आपका बहुत बहुत धन्यवाद", "धन्यवाद = genuine thanks (1,159 occ); 'देखने के लिए धन्यवाद' = 0"),
    ("ऑडियो वीडियो कैसेट", "वीडियो = cassette catalog, not 'watch the video'"),
    ("दुख शेयर करना", "शेयर = share sorrow, not a YouTube CTA"),
    ("समझ आई कि नहीं", "bare आई = the verb 'came' (6,252 occ), never 'eye'"),
    ("आपकी ऊर्जा को आपकी एनर्जी को", "ऊर्जा is native Sanskrit, deliberately glossed by एनर्जी"),
    ("मैं आया।दूसरा वाक्य।", "a glued danda is NOT rewritten — Tantivy already splits on it"),
    ("जागरण—enlightened", "an em-dash gloss is NOT rewritten — U+2014 already splits"),
])
def test_anti_rule_surface_form_survives_normalization(rules, text, why):
    assert nt.normalize(text, rules).text == text, why


def test_english_quotation_is_never_rewritten(rules):
    """43.9% of Latin `watch` and 47.5% of `seer` sit inside verbatim English
    speech. Cross-script bridging is a query-time synonym table, never a rewrite."""
    text = "you are not a mind, you are a seer of it — keep watching the master"
    assert nt.normalize(text, rules).text == text


# ---- the shipped rules file ---------------------------------------------


def test_shipped_rules_file_loads_and_is_whole_token():
    r = nt.load_rules(RULES_JSON)
    assert r.loop_min_run >= nt.LOOP_MIN_RUN
    assert r.loop_keep >= nt.LOOP_KEEP
    assert "सब्सक्राइब" in r.hallucination_tokens
    assert "और" not in r.hallucination_tail, "the tail must never cross और"
    # No key may contain whitespace: map_tokens matches one token at a time, so a
    # two-token target could never fire and would silently do nothing.
    for table in (r.loanword_tokens, r.native_tokens, r.spelling_scatter):
        for src, tgt in table.items():
            assert " " not in src, src
            assert " " not in tgt, tgt


def test_shipped_rules_repair_the_headline_defects():
    r = nt.load_rules(RULES_JSON)
    assert r.loanword_tokens["मEDITATION"] == "मेडिटेशन"   # 1,498 occ / 417 files
    assert r.native_tokens["रishi"] == "रिशी"              # 578 occ / 251 files
    assert r.native_tokens["Kumarों"] == "कुमारों"          # 18 occ / 11 files
    assert r.spelling_scatter["मीडिटेशन"] == "मेडिटेशन"     # 501 occ


def test_no_shipped_rule_maps_a_valid_devanagari_word_to_another():
    """A rewrite is allowed only where the source has no valid reading: it is
    script-mixed, or it is a known misspelling of its own target."""
    r = nt.load_rules(RULES_JSON)
    for src in list(r.loanword_tokens) + list(r.native_tokens):
        assert nt.is_script_mixed(src), f"{src} is not script-mixed; do not rewrite it"
    for src, tgt in r.spelling_scatter.items():
        assert not nt.is_script_mixed(src)
        assert src != tgt


def test_shipped_rules_have_no_flagged_names():
    """Personal names whose Devanagari spelling cannot be verified from the corpus
    are flagged for a human, never auto-mapped. `Vishwas`->विश्वास would collide
    with the common noun (2,105 occ)."""
    r = nt.load_rules(RULES_JSON)
    for flagged in ("महारishi", "रishiRai", "कishkinda", "दधichi", "अरindo"):
        assert flagged not in r.native_tokens
        assert flagged not in r.loanword_tokens


def test_rules_file_is_valid_json_with_notes():
    raw = json.loads(RULES_JSON.read_text(encoding="utf-8"))
    assert "_notes" in raw and "substring" in raw["_notes"]
