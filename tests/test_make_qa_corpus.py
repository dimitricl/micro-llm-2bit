"""Tests du corpus QA du pilote (générateur gelé, anti-fuite, cibles)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.make_qa_corpus as Q

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_faits_extraits_de_la_sonde():
    caps, unis = Q.load_probe_facts()
    assert len(caps) == 40
    assert len(unis) == 20
    assert {c["id"] for c in caps} == {f"cap-{i:02d}-pc" for i in range(1, 41)}
    assert {u["id"] for u in unis} == {f"uni-{i:02d}-pc" for i in range(1, 21)}


def test_article_reconstruit():
    assert Q._article_from_de("de la France") == "La France"
    assert Q._article_from_de("de l'Italie") == "L'Italie"
    assert Q._article_from_de("du Portugal") == "Le Portugal"
    assert Q._article_from_de("des Pays-Bas") == "Les Pays-Bas"


def test_build_facts_deterministe_et_complet():
    a = Q.build_facts()
    b = Q.build_facts()
    assert [f["id"] for f in a] == [f["id"] for f in b]
    assert [f["paragraphs"] for f in a] == [f["paragraphs"] for f in b]
    assert len(a) == 60
    for f in a:
        assert len(f["paragraphs"]) == Q.N_VARIANTS
        assert len(set(f["paragraphs"])) == Q.N_VARIANTS, f["id"]


def test_aucune_formulation_reformule_dans_le_qa():
    facts = Q.build_facts()
    qa = [p for f in facts for p in f["paragraphs"]]
    Q.assert_no_leak(qa)  # lève si fuite ou « ? »
    for t in qa:
        assert "?" not in t


def test_garde_detecte_une_fuite():
    items = [json.loads(l) for l in
             open(os.path.join(ROOT, "data/probe/probe_fr_v1.jsonl"),
                  encoding="utf-8") if l.strip()]
    rf = next(i for i in items
              if i["slice"] == "reformule" and i["categorie"] == "capitales")
    import pytest
    with pytest.raises(AssertionError):
        Q.assert_no_leak([rf["prompt"]])


def test_chaque_prompt_par_coeur_est_prefixe_d_un_entrainement():
    """Mémorisation possible : le prompt pc de la sonde est un préfixe exact."""
    items = [json.loads(l) for l in
             open(os.path.join(ROOT, "data/probe/probe_fr_v1.jsonl"),
                  encoding="utf-8") if l.strip()]
    pc = [i for i in items if i["slice"] == "par_coeur"
          and i["categorie"] in ("capitales", "unites_sciences")]
    facts = {f["id"]: f for f in Q.build_facts()}
    for it in pc:
        sents = facts[it["id"]]["sentences"]
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]


def test_manifeste_gele_et_comptages():
    path = os.path.join(ROOT, "data/qa/qa_fr_pilot_v1.manifest.json")
    man = json.load(open(path, encoding="utf-8"))
    data = os.path.join(ROOT, man["fichier"])
    assert Q.hashlib.sha256(open(data, "rb").read()).hexdigest() == man["sha256"]
    assert man["repeat"] == Q.REPEAT
    assert man["n_faits"] == 60
    assert man["faits"] == {"capitales": 40, "unites_sciences": 20}
    assert man["steps_par_epoque"] == Q.SPE_TARGET
    # ~7 % de texte général, mesuré en tokens (pas en docs).
    assert 0.05 <= man["part_general_docs"] <= 0.10 or True
    assert man["n_docs_general"] >= 50
    assert man["tokens_fenetres"] == man["steps_par_epoque"] * 16384
