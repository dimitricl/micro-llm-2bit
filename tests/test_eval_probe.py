"""Tests du harnais de sonde (probe FR) : normalisation, alias, EOS, agrégats.

Écrits AVANT l'implémentation (TDD).
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tools.eval_probe as P
import tools.make_probe as MP


# ---------------------------------------------------------------------------
# Normalisation (règle documentée)
# ---------------------------------------------------------------------------

def test_normalize_minuscules_et_espaces():
    assert P.normalize("  Paris  ") == "paris"
    assert P.normalize("NEW   DELHI") == "new delhi"


def test_normalize_accents():
    assert P.normalize("Pékin") == "pekin"
    assert P.normalize("SÉOUL") == "seoul"
    assert P.normalize("À Athènes") == "a athenes"


def test_normalize_ponctuation():
    assert P.normalize("Paris.") == "paris"
    assert P.normalize("Paris, en France") == "paris en france"
    assert P.normalize("9,8") == "9 8"
    assert P.normalize("9.8") == "9 8"
    assert P.normalize("l'eau") == "l eau"


def test_normalize_vide():
    assert P.normalize("") == ""
    assert P.normalize("...") == ""


# ---------------------------------------------------------------------------
# Matching par alias (frontières de tokens)
# ---------------------------------------------------------------------------

def test_hit_alias_simple():
    assert P.is_hit("La capitale est Paris.", ["Paris"]) is True
    assert P.is_hit("la capitale est paris", ["Paris"]) is True
    assert P.is_hit("c'est PÉKIN.", ["Pékin"]) is True


def test_miss_sans_alias():
    assert P.is_hit("La capitale est Lyon.", ["Paris"]) is False
    assert P.is_hit("", ["Paris"]) is False


def test_frontiere_de_token_numerique():
    """« 100 » ne doit PAS matcher dans « 1000 » (pas de sous-chaîne nue)."""
    assert P.is_hit("il fait 100 degrés", ["100"]) is True
    assert P.is_hit("il fait 1000 degrés", ["100"]) is False
    assert P.is_hit("environ 2100 mètres", ["100"]) is False


def test_alias_multi_mots():
    assert P.is_hit("la ville de New Delhi est", ["New Delhi"]) is True
    assert P.is_hit("la ville de NewDelhi est", ["New Delhi"]) is False


def test_alias_liste_un_seul_suffit():
    assert P.is_hit("la lumière est captée", ["chlorophylle", "lumière"]) is True
    assert P.is_hit("rien de pertinent", ["chlorophylle", "lumière"]) is False


def test_alias_vide_ignore():
    assert P.is_hit("paris", ["", "Paris"]) is True
    assert P.is_hit("paris", [""]) is False


# ---------------------------------------------------------------------------
# Collapse EOS
# ---------------------------------------------------------------------------

def test_eos_collapse_seuil():
    assert P.eos_collapse(0) is True
    assert P.eos_collapse(1) is True
    assert P.eos_collapse(4) is True
    assert P.eos_collapse(5) is False
    assert P.eos_collapse(60) is False


def test_item_eos_collapse_dans_le_score():
    """Un item à 1 token est compté EOS-collapse ET miss."""
    rec = P.score_one(
        {"id": "x", "categorie": "dates", "slice": "reformule",
         "perimetre": "wiki_general", "prompt": "Quand ?",
         "reponses_acceptees": ["1789"]},
        gen_text="", n_tokens=1)
    assert rec["hit"] is False
    assert rec["eos_collapse"] is True


# ---------------------------------------------------------------------------
# Agrégats
# ---------------------------------------------------------------------------

def _rec(id_, cat, sl, per, hit, ntok=20):
    return {"id": id_, "categorie": cat, "slice": sl, "perimetre": per,
            "prompt": "p", "gen": "", "n_tokens": ntok,
            "hit": hit, "eos_collapse": ntok < 5}


def test_aggregate_par_slice_categorie_perimetre():
    recs = [
        _rec("a", "capitales", "par_coeur", "domaine_etroit", True),
        _rec("b", "capitales", "par_coeur", "domaine_etroit", False),
        _rec("c", "capitales", "reformule", "domaine_etroit", False, ntok=2),
        _rec("d", "dates", "reformule", "wiki_general", True),
    ]
    agg = P.aggregate(recs)
    assert agg["global"]["n"] == 4
    assert agg["global"]["hits"] == 2
    assert agg["global"]["taux"] == 0.5
    assert agg["global"]["taux_eos_collapse"] == 0.25
    assert agg["par_slice"]["par_coeur"]["taux"] == 0.5
    assert agg["par_slice"]["reformule"]["taux"] == 0.5
    assert agg["par_categorie"]["capitales"]["n"] == 3
    assert agg["par_categorie"]["capitales"]["hits"] == 1
    assert agg["par_perimetre"]["domaine_etroit"]["n"] == 3
    assert agg["par_perimetre"]["wiki_general"]["n"] == 1
    cell = agg["par_categorie_x_slice"]["capitales"]["reformule"]
    assert cell["n"] == 1 and cell["hits"] == 0


def test_aggregate_vide_ne_divise_pas_par_zero():
    agg = P.aggregate([])
    assert agg["global"]["n"] == 0
    assert agg["global"]["taux"] == 0.0


# ---------------------------------------------------------------------------
# Sonde gelée : schéma + déterminisme
# ---------------------------------------------------------------------------

def test_probe_generee_est_deterministe_et_conforme():
    a = MP.build()
    b = MP.build()
    assert a == b, "build() doit être déterministe"
    assert len(a) >= 200
    for it in a:
        assert set(it) == {"id", "categorie", "slice", "perimetre", "prompt",
                           "reponses_acceptees", "note"}
        assert it["categorie"] in ("capitales", "dates", "definitions",
                                   "unites_sciences")
        assert it["slice"] in ("par_coeur", "reformule")
        assert it["perimetre"] in ("domaine_etroit", "wiki_general")
        assert isinstance(it["reponses_acceptees"], list)
        assert it["reponses_acceptees"]
    # 50/50 par catégorie.
    for cat in ("capitales", "dates", "definitions", "unites_sciences"):
        sub = [i for i in a if i["categorie"] == cat]
        pc = sum(1 for i in sub if i["slice"] == "par_coeur")
        assert pc == len(sub) - pc


def test_manifeste_correspond_au_jsonl_git():
    root = os.path.join(os.path.dirname(__file__), "..")
    man = json.load(open(os.path.join(root, "data/probe/probe_fr_v1.manifest.json"),
                         encoding="utf-8"))
    path = os.path.join(root, man["fichier"])
    assert P.sha256_file(path) == man["sha256"], \
        "la sonde a été modifiée après gel (hash != manifeste)"
    assert man["n_items"] == len(MP.build())


def test_eval_probe_refuse_une_sonde_modifiee(tmp_path):
    """Anti-ajustement a posteriori : hash != manifeste -> refus bruyant."""
    probe = tmp_path / "probe.jsonl"
    probe.write_text('{"id": "x"}\n', encoding="utf-8")
    man = tmp_path / "man.json"
    man.write_text(json.dumps({"sha256": "0" * 64, "fichier": "probe.jsonl",
                               "n_items": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        P.load_probe(str(probe), manifest_path=str(man))
