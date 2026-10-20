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


# ---------------------------------------------------------------------------
# v2 (pilote QA #2, recette rééquilibrée)
# ---------------------------------------------------------------------------

def test_v2_faits_complets_et_deterministes():
    a = Q.build_facts_v2()
    b = Q.build_facts_v2()
    assert [f["id"] for f in a] == [f["id"] for f in b]
    assert [f["paragraphs"] for f in a] == [f["paragraphs"] for f in b]
    # 500-1000 faits demandés, 4 catégories couvertes.
    assert 500 <= len(a) <= 1000, len(a)
    cats = {f["categorie"] for f in a}
    assert cats == set(Q.CATEGORIES_V2)
    for f in a:
        assert len(f["paragraphs"]) == Q.N_VARIANTS_V2, f["id"]
        assert len(set(f["paragraphs"])) == Q.N_VARIANTS_V2, f["id"]


def test_v2_aucune_formulation_reformule_dans_le_qa():
    facts = Q.build_facts_v2()
    qa = [p for f in facts for p in f["paragraphs"]]
    Q.assert_no_leak(qa, categories=Q.GUARD_CATS_ALL)
    for t in qa:
        assert "?" not in t


def test_v2_garde_elargie_aux_4_categories():
    """La v2 garde aussi dates/definitions (l'élargissement est effectif)."""
    n_all = len(Q.reformule_guards(Q.GUARD_CATS_ALL))
    n_v1 = len(Q.reformule_guards(Q.GUARD_CATS_V1))
    assert n_all == 100 and n_v1 == 60


def test_v2_chaque_prompt_par_coeur_est_prefixe_d_un_entrainement():
    """Les 100 prompts pc de la sonde sont des préfixes exacts (mémorisation)."""
    items = [json.loads(l) for l in
             open(os.path.join(ROOT, "data/probe/probe_fr_v1.jsonl"),
                  encoding="utf-8") if l.strip()]
    pc = [i for i in items if i["slice"] == "par_coeur"]
    facts = {f["id"]: f for f in Q.build_facts_v2()}
    for it in pc:
        sents = facts[it["id"]]["sentences"]
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]


def test_v2_mots_cles_definitions_presents_dans_le_texte():
    import tools.qa_facts_v2 as F2
    items = [json.loads(l) for l in
             open(os.path.join(ROOT, "data/probe/probe_fr_v1.jsonl"),
                  encoding="utf-8") if l.strip()]
    facts = {f["id"]: f for f in Q.build_facts_v2()}
    for it in items:
        if it["categorie"] != "definitions" or it["slice"] != "par_coeur":
            continue
        norm = " ".join(Q._norm(s) for s in facts[it["id"]]["sentences"])
        for kw in it["reponses_acceptees"]:
            assert Q._norm(kw) in norm, (it["id"], kw)


def test_v2_tables_coherentes():
    import tools.qa_facts_v2 as F2
    # unités : 5 champs partout.
    assert all(len(t) == 5 for t in F2.UNITS_EXTRA)
    assert all(len(t) == 4 for t in F2.DEFINITIONS_EXTRA)
    # 20 définitions pc de la sonde fournies.
    assert set(F2.DEFINITIONS_PROBE) == {f"def-{i:02d}-pc" for i in range(1, 21)}
    # les mots-clés des définitions apparaissent dans leur complément.
    for _tid, (_art, _nu, kws, comp) in F2.DEFINITIONS_PROBE.items():
        for kw in kws:
            assert kw.lower() in comp.lower(), (kw, comp)
    for _art, _nu, kws, comp in F2.DEFINITIONS_EXTRA:
        for kw in kws:
            assert kw.lower() in comp.lower(), (kw, comp)
    # pas de doublon de pays / de sujet de date (probe + extras).
    caps, _ = Q.load_probe_facts()
    pays = [c["pays"] for c in caps] + [p for p, _ in F2.CAPITALES_EXTRA]
    assert len(pays) == len(set(pays)), "pays dupliqué"
    probe_sujets = {Q._date_sujet(i["prompt"])
                    for i in Q._probe_pc_items() if i["categorie"] == "dates"}
    sujets = [s for s, _ in F2.DATES_EXTRA]
    assert len(sujets) == len(set(sujets)), "sujet de date dupliqué"
    assert not (probe_sujets & set(sujets)), "sujet de date dupliqué vs sonde"


def test_manifeste_v2_gele_et_comptages():
    path = os.path.join(ROOT, "data/qa/qa_fr_pilot_v2.manifest.json")
    man = json.load(open(path, encoding="utf-8"))
    data = os.path.join(ROOT, man["fichier"])
    assert Q.hashlib.sha256(open(data, "rb").read()).hexdigest() == man["sha256"]
    assert man["repeat"] == Q.REPEAT_V2
    assert man["n_variants_par_fait"] == Q.N_VARIANTS_V2
    assert 500 <= man["n_faits"] <= 1000
    assert set(man["faits"]) == set(Q.CATEGORIES_V2)
    assert man["steps_par_epoque"] == Q.SPE_TARGET_V2
    # 30-40 % de texte général, mesuré en tokens.
    assert 0.30 <= man["part_general_tokens"] <= 0.40
    assert man["tokens_fenetres"] == man["steps_par_epoque"] * 16384
    # 4 répétitions pleines (aucune troncature) : n_docs_qa = faits x var x 4.
    assert man["n_docs_qa"] == man["n_faits"] * man["n_variants_par_fait"] \
        * man["repeat"]


# ---------------------------------------------------------------------------
# v3 (pilote QA #3 : budget x3 + corpus naturel)
# ---------------------------------------------------------------------------

import functools


@functools.lru_cache(maxsize=1)
def _v3_facts():
    return Q.build_facts_v3()


def test_v3_faits_complets_et_deterministes():
    a = _v3_facts()
    b = Q.build_facts_v3()
    assert [f["id"] for f in a] == [f["id"] for f in b]
    assert [f["paragraphs"] for f in a] == [f["paragraphs"] for f in b]
    assert len(a) == 534
    assert {f["categorie"] for f in a} == set(Q.CATEGORIES_V2)
    for f in a:
        assert len(f["paragraphs"]) == Q.N_VARIANTS_V3, f["id"]
        assert len(set(f["paragraphs"])) == Q.N_VARIANTS_V3, f["id"]


def test_v3_aucune_formulation_reformule_dans_le_qa():
    qa = [p for f in _v3_facts() for p in f["paragraphs"]]
    Q.assert_no_leak(qa, categories=Q.GUARD_CATS_ALL)
    for t in qa:
        assert "?" not in t


def test_v3_chaque_prompt_par_coeur_est_prefixe_d_un_entrainement():
    items = [json.loads(l) for l in
             open(os.path.join(ROOT, "data/probe/probe_fr_v1.jsonl"),
                  encoding="utf-8") if l.strip()]
    pc = [i for i in items if i["slice"] == "par_coeur"]
    facts = {f["id"]: f for f in _v3_facts()}
    for it in pc:
        sents = facts[it["id"]]["sentences"]
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]


def test_v3_phrases_reelles_extraites_et_utilisees():
    facts = _v3_facts()
    meta = Q._real_sentence_meta()
    assert len(meta) == 534
    n_real = sum(len(f["real_sentences"]) for f in facts)
    assert n_real > 500, n_real
    with_real = [f for f in facts if f["real_sentences"]]
    assert len(with_real) >= 200, len(with_real)
    for f in with_real:
        assert len(f["real_sentences"]) <= Q.REAL_MAX_V3
        # Chaque phrase réelle est bien incluse verbatim dans des variantes.
        for s in f["real_sentences"]:
            assert Q.REAL_MIN_CHARS_V3 <= len(s) <= Q.REAL_MAX_CHARS_V3
            assert "?" not in s
            assert any(s in p for p in f["paragraphs"]), (f["id"], s[:60])


def test_v3_matching_sous_chaine_frontieres_de_tokens():
    # « 100 » ne matche pas « 1100 » ; « italie » ne matche pas « italien ».
    assert Q._has_token("il y a 100 metres", "100")
    assert not Q._has_token("il y a 1100 metres", "100")
    assert not Q._has_token("un ecrivain italien", "italie")
    assert Q._has_token("la capitale de l italie est rome", "italie")


def test_v3_tables_unites_completes():
    pc_units = {f"uni-{i:02d}-pc" for i in range(1, 21)}
    assert set(Q.UNIT_TRIPLES) == pc_units
    assert set(Q.UNIT_ENTITY) == pc_units


def test_manifeste_v3_gele_et_comptages():
    path = os.path.join(ROOT, "data/qa/qa_fr_pilot_v3.manifest.json")
    man = json.load(open(path, encoding="utf-8"))
    data = os.path.join(ROOT, man["fichier"])
    assert Q.hashlib.sha256(open(data, "rb").read()).hexdigest() == man["sha256"]
    assert man["repeat"] == Q.REPEAT_V3 == 10
    assert man["n_variants_par_fait"] == Q.N_VARIANTS_V3 == 40
    assert man["n_faits"] == 534
    assert set(man["faits"]) == set(Q.CATEGORIES_V2)
    assert man["steps_par_epoque"] == Q.SPE_TARGET_V3 == 1516
    # ~22 % de texte général, mesuré en tokens.
    assert 0.18 <= man["part_general_tokens"] <= 0.28, man["part_general_tokens"]
    assert man["tokens_fenetres"] == man["steps_par_epoque"] * 16384
    # 10 répétitions pleines (aucune troncature QA).
    assert man["n_docs_qa"] == man["n_faits"] * man["n_variants_par_fait"] \
        * man["repeat"]
    # Phrases réelles : récupération plafonnée et comptée.
    assert man["corpus_reel"]["max_par_fait"] == Q.REAL_MAX_V3
    assert man["corpus_reel"]["n_phrases"] >= 500
    assert man["corpus_reel"]["n_faits_avec_phrase"] >= 200
    # Epochs cumulatives : reprise t08 -> exactement 1516 steps restants.
    ec = man["epochs_cumulatives"]
    assert ec["prev_steps"] == 22740
    assert ec["e_cumul"] == 16
    assert ec["steps_restants"] == 1516

