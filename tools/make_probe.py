"""Sonde FR gelée v1 : mesure « récite » vs « généralise » (CPU, déterministe).

200 items = 4 catégories × 2 slices (par_coeur / reformule), 50/50 par
catégorie. Chaque fait produit DEUX items : une formulation proche des
templates du corpus d'entraînement (`par_coeur`) et la même question
tournée autrement (`reformule`), jamais vue. Le GAP entre les deux slices
est le diagnostic : un modèle qui récite ses templates marque en
par_coeur et s'effondre en reformule.

Périmètres (décision projet A/B) :
  - domaine_etroit : géographie / capitales (catégorie `capitales`).
  - wiki_general   : le reste (`dates`, `definitions`, `unites_sciences`).

Sorties :
  data/probe/probe_fr_v1.jsonl            (gelé, committé)
  data/probe/probe_fr_v1.manifest.json    (SHA256 + comptages, committé)

Le JSONL est GELÉ avant toute mesure : aucun item ajouté/retiré après
les scores (anti-ajustement a posteriori). Même esprit que val_E_frozen.

Usage : .venv/bin/python tools/make_probe.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR = os.path.join(ROOT, "data", "probe")
OUT_JSONL = os.path.join(OUT_DIR, "probe_fr_v1.jsonl")
OUT_MANIFEST = os.path.join(OUT_DIR, "probe_fr_v1.manifest.json")

VERSION = "probe_fr_v1"

# ---------------------------------------------------------------------------
# Faits. `de` porte l'article contracté pour un usage grammatical correct.
# ---------------------------------------------------------------------------

# (forme avec article, forme « de … », capitale)
CAPITALES = [
    ("La France", "de la France", "Paris"),
    ("L'Italie", "de l'Italie", "Rome"),
    ("L'Espagne", "de l'Espagne", "Madrid"),
    ("L'Allemagne", "de l'Allemagne", "Berlin"),
    ("Le Portugal", "du Portugal", "Lisbonne"),
    ("La Belgique", "de la Belgique", "Bruxelles"),
    ("La Suisse", "de la Suisse", "Berne"),
    ("L'Autriche", "de l'Autriche", "Vienne"),
    ("La Grèce", "de la Grèce", "Athènes"),
    ("La Pologne", "de la Pologne", "Varsovie"),
    ("La Russie", "de la Russie", "Moscou"),
    ("La Chine", "de la Chine", "Pékin"),
    ("Le Japon", "du Japon", "Tokyo"),
    ("L'Inde", "de l'Inde", "New Delhi"),
    ("Le Brésil", "du Brésil", "Brasilia"),
    ("L'Argentine", "de l'Argentine", "Buenos Aires"),
    ("Le Canada", "du Canada", "Ottawa"),
    ("Le Mexique", "du Mexique", "Mexico"),
    ("L'Égypte", "de l'Égypte", "Le Caire"),
    ("Le Maroc", "du Maroc", "Rabat"),
    ("L'Algérie", "de l'Algérie", "Alger"),
    ("La Tunisie", "de la Tunisie", "Tunis"),
    ("Le Sénégal", "du Sénégal", "Dakar"),
    ("Le Nigeria", "du Nigeria", "Abuja"),
    ("Le Kenya", "du Kenya", "Nairobi"),
    ("L'Afrique du Sud", "de l'Afrique du Sud", "Pretoria"),
    ("L'Australie", "de l'Australie", "Canberra"),
    ("La Norvège", "de la Norvège", "Oslo"),
    ("La Suède", "de la Suède", "Stockholm"),
    ("Le Danemark", "du Danemark", "Copenhague"),
    ("La Finlande", "de la Finlande", "Helsinki"),
    ("L'Irlande", "de l'Irlande", "Dublin"),
    ("Les Pays-Bas", "des Pays-Bas", "Amsterdam"),
    ("La Turquie", "de la Turquie", "Ankara"),
    ("L'Iran", "de l'Iran", "Téhéran"),
    ("La Corée du Sud", "de la Corée du Sud", "Séoul"),
    ("Le Viêt Nam", "du Viêt Nam", "Hanoï"),
    ("La Thaïlande", "de la Thaïlande", "Bangkok"),
    ("L'Ukraine", "de l'Ukraine", "Kiev"),
    ("La Roumanie", "de la Roumanie", "Bucarest"),
]

# (sujet, annee)
DATES = [
    ("la Révolution française", "1789"),
    ("la prise de la Bastille", "1789"),
    ("le sacre de Napoléon", "1804"),
    ("la bataille de Waterloo", "1815"),
    ("la chute de Constantinople", "1453"),
    ("la découverte de l'Amérique", "1492"),
    ("la mort de Louis XIV", "1715"),
    ("la Révolution russe", "1917"),
    ("la fin de la Première Guerre mondiale", "1918"),
    ("la fin de la Seconde Guerre mondiale", "1945"),
    ("la libération de Paris", "1944"),
    ("la bataille de Verdun", "1916"),
    ("le traité de Versailles", "1919"),
    ("le traité de Rome", "1957"),
    ("la chute du mur de Berlin", "1989"),
    ("Mai 68", "1968"),
    ("la création de l'ONU", "1945"),
    ("le premier pas sur la Lune", "1969"),
    ("la Révolution de 1848", "1848"),
    ("l'abdication de Napoléon", "1814"),
]

# (forme nominale avec article, terme nu, alias acceptés)
DEFINITIONS = [
    ("la photosynthèse", "photosynthèse", ["chlorophylle", "lumière"]),
    ("la gravité", "gravité", ["attraction", "masse"]),
    ("la monarchie", "monarchie", ["roi"]),
    ("la démocratie", "démocratie", ["peuple"]),
    ("l'ADN", "ADN", ["génétique", "acide"]),
    ("une métaphore", "métaphore", ["comparaison"]),
    ("une péninsule", "péninsule", ["mer"]),
    ("un archipel", "archipel", ["îles"]),
    ("un fleuve", "fleuve", ["mer"]),
    ("une rivière", "rivière", ["affluent"]),
    ("le climat", "climat", ["température"]),
    ("l'économie", "économie", ["production", "richesse"]),
    ("la philosophie", "philosophie", ["pensée"]),
    ("un roman", "roman", ["fiction", "personnages"]),
    ("une symphonie", "symphonie", ["orchestre"]),
    ("un vaccin", "vaccin", ["immunité", "maladie"]),
    ("l'inflation", "inflation", ["prix"]),
    ("un écosystème", "écosystème", ["espèces", "milieu"]),
    ("une altitude", "altitude", ["mer"]),
    ("la génétique", "génétique", ["gènes", "hérédité"]),
]

# (prompt par_coeur, prompt reformule, alias acceptés)
UNITES = [
    ("L'eau bout à", "À quelle température l'eau bout-elle ?", ["100"]),
    ("L'eau gèle à", "À quelle température l'eau gèle-t-elle ?", ["0"]),
    ("Un kilomètre vaut", "Combien de mètres contient un kilomètre ?", ["1000"]),
    ("Un mètre contient", "Combien de centimètres contient un mètre ?", ["100"]),
    ("Une heure dure", "Combien de minutes dure une heure ?", ["60"]),
    ("Un kilogramme vaut", "Combien de grammes contient un kilogramme ?", ["1000"]),
    ("Un litre vaut", "Combien de millilitres contient un litre ?", ["1000"]),
    ("La vitesse de la lumière est de",
     "Quelle est la vitesse de la lumière ?", ["300000", "299792"]),
    ("Le corps humain compte", "Combien d'os compte le corps humain ?", ["206"]),
    ("La température du corps humain est de",
     "Quelle est la température normale du corps humain ?", ["37"]),
    ("Le système solaire compte", "Combien de planètes compte le système solaire ?",
     ["8", "huit"]),
    ("Une journée compte", "Combien de secondes compte une journée ?", ["86400"]),
    ("La Terre tourne autour du Soleil en",
     "En combien de jours la Terre tourne-t-elle autour du Soleil ?", ["365"]),
    ("Le symbole chimique de l'or est", "Quel est le symbole chimique de l'or ?",
     ["au"]),
    ("La molécule d'eau est", "Quelle est la formule chimique de l'eau ?", ["h2o"]),
    ("L'accélération de la pesanteur est de",
     "Quelle est l'accélération de la pesanteur sur Terre ?", ["9,8", "9.8"]),
    ("Le point d'ébullition de l'azote est de",
     "À quelle température l'azote bout-il ?", ["-196"]),
    ("Le nombre de chromosomes humains est de",
     "Combien de chromosomes compte une cellule humaine ?", ["46"]),
    ("Une année compte", "Combien de jours compte une année non bissextile ?",
     ["365"]),
    ("La température d'ébullition de l'eau est de",
     "À quelle température l'eau bout-elle en Fahrenheit ?",
     ["212"]),]

TEMPLATES = {
    "capitales": {
        "par_coeur": ["La capitale {de} est",
                      "{pays} est un pays dont la capitale est"],
        "reformule": ["Quelle est la capitale {de} ?",
                      "Quelle ville est la capitale {de} ?"],
    },    "dates": {
        "par_coeur": ["{sujet} a eu lieu en", "{sujet} date de"],
        "reformule": ["En quelle année a eu lieu {sujet} ?",
                      "Quand a eu lieu {sujet} ?"],
    },
    "definitions": {
        "par_coeur": ["{terme} est", "Le mot « {nu} » désigne"],
        "reformule": ["Que signifie « {nu} » ?",
                      "Comment définit-on {terme} ?"],
    },
}

CATEGORIES = ["capitales", "dates", "definitions", "unites_sciences"]
PERIMETRE = {
    "capitales": "domaine_etroit",
    "dates": "wiki_general",
    "definitions": "wiki_general",
    "unites_sciences": "wiki_general",
}


def _capitales_items():
    out = []
    for i, (pays, de, cap) in enumerate(CAPITALES, start=1):
        out.append({
            "id": f"cap-{i:02d}-pc",
            "categorie": "capitales",
            "slice": "par_coeur",
            "perimetre": PERIMETRE["capitales"],
            "prompt": TEMPLATES["capitales"]["par_coeur"][i % 2].format(
                pays=pays, de=de),
            "reponses_acceptees": [cap],
            "note": f"capitale officielle {de}",
        })
        out.append({
            "id": f"cap-{i:02d}-rf",
            "categorie": "capitales",
            "slice": "reformule",
            "perimetre": PERIMETRE["capitales"],
            "prompt": TEMPLATES["capitales"]["reformule"][i % 2].format(
                pays=pays, de=de),
            "reponses_acceptees": [cap],
            "note": f"même fait que cap-{i:02d}-pc, formulation tournée",
        })
    return out


def _dates_items():
    out = []
    for i, (sujet, an) in enumerate(DATES, start=1):
        out.append({
            "id": f"dat-{i:02d}-pc",
            "categorie": "dates",
            "slice": "par_coeur",
            "perimetre": PERIMETRE["dates"],
            "prompt": TEMPLATES["dates"]["par_coeur"][i % 2].format(sujet=sujet),
            "reponses_acceptees": [an],
            "note": f"date de {sujet}",
        })
        out.append({
            "id": f"dat-{i:02d}-rf",
            "categorie": "dates",
            "slice": "reformule",
            "perimetre": PERIMETRE["dates"],
            "prompt": TEMPLATES["dates"]["reformule"][i % 2].format(sujet=sujet),
            "reponses_acceptees": [an],
            "note": f"même fait que dat-{i:02d}-pc, formulation tournée",
        })
    return out


def _definitions_items():
    out = []
    for i, (terme, nu, alias) in enumerate(DEFINITIONS, start=1):
        out.append({
            "id": f"def-{i:02d}-pc",
            "categorie": "definitions",
            "slice": "par_coeur",
            "perimetre": PERIMETRE["definitions"],
            "prompt": TEMPLATES["definitions"]["par_coeur"][i % 2].format(
                terme=terme, nu=nu),
            "reponses_acceptees": list(alias),
            "note": f"mot-clé définitionnel de {terme}",
        })
        out.append({
            "id": f"def-{i:02d}-rf",
            "categorie": "definitions",
            "slice": "reformule",
            "perimetre": PERIMETRE["definitions"],
            "prompt": TEMPLATES["definitions"]["reformule"][i % 2].format(
                terme=terme, nu=nu),
            "reponses_acceptees": list(alias),
            "note": f"même fait que def-{i:02d}-pc, formulation tournée",
        })
    return out


def _unites_items():
    out = []
    for i, (pc, rf, alias) in enumerate(UNITES, start=1):
        out.append({
            "id": f"uni-{i:02d}-pc",
            "categorie": "unites_sciences",
            "slice": "par_coeur",
            "perimetre": PERIMETRE["unites_sciences"],
            "prompt": pc,
            "reponses_acceptees": list(alias),
            "note": "constante / conversion standard",
        })
        out.append({
            "id": f"uni-{i:02d}-rf",
            "categorie": "unites_sciences",
            "slice": "reformule",
            "perimetre": PERIMETRE["unites_sciences"],
            "prompt": rf,
            "reponses_acceptees": list(alias),
            "note": f"même fait que uni-{i:02d}-pc, formulation tournée",
        })
    return out


def build() -> list[dict]:
    items = (_capitales_items() + _dates_items()
             + _definitions_items() + _unites_items())
    # Garde-fous de gel : unicité d'id et de prompt, 50/50 par catégorie.
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "ids dupliqués"
    prompts = [it["prompt"] for it in items]
    assert len(prompts) == len(set(prompts)), "prompts dupliqués"
    for cat in CATEGORIES:
        sub = [it for it in items if it["categorie"] == cat]
        pc = sum(1 for it in sub if it["slice"] == "par_coeur")
        rf = sum(1 for it in sub if it["slice"] == "reformule")
        assert pc == rf, f"{cat} : {pc} par_coeur vs {rf} reformule"
        assert pc >= 20, f"{cat} : {pc} items par slice"
    for it in items:
        assert it["categorie"] in CATEGORIES, it
        assert it["slice"] in ("par_coeur", "reformule"), it
        assert it["perimetre"] in ("domaine_etroit", "wiki_general"), it
        assert it["reponses_acceptees"], it
        assert all(isinstance(r, str) and r for r in it["reponses_acceptees"]), it
    return items


def main() -> None:
    items = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    blob = open(OUT_JSONL, "rb").read()
    sha = hashlib.sha256(blob).hexdigest()

    def count(pred):
        return sum(1 for it in items if pred(it))

    manifest = {
        "version": VERSION,
        "fichier": os.path.relpath(OUT_JSONL, ROOT),
        "sha256": sha,
        "octets": len(blob),
        "n_items": len(items),
        "categories": {
            c: {
                "n": count(lambda it, c=c: it["categorie"] == c),
                "par_coeur": count(lambda it, c=c: it["categorie"] == c
                                   and it["slice"] == "par_coeur"),
                "reformule": count(lambda it, c=c: it["categorie"] == c
                                   and it["slice"] == "reformule"),
            } for c in CATEGORIES
        },
        "slices": {
            s: count(lambda it, s=s: it["slice"] == s)
            for s in ("par_coeur", "reformule")
        },
        "perimetres": {
            p: count(lambda it, p=p: it["perimetre"] == p)
            for p in ("domaine_etroit", "wiki_general")
        },
        "regle_de_gel": (
            "Sonde figée AVANT toute mesure : hash SHA256 du JSONL ci-dessus. "
            "Aucun item ajouté, retiré ou modifié après le premier score ; "
            "toute évolution passe par probe_fr_v2 (nouveau fichier)."),
        "generateur": os.path.relpath(os.path.abspath(__file__), ROOT),
        "generateur_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"[probe] {len(items)} items -> {os.path.relpath(OUT_JSONL, ROOT)}")
    print(f"[probe] sha256 {sha}")
    print(f"[probe] manifeste -> {os.path.relpath(OUT_MANIFEST, ROOT)}")
    for c in CATEGORIES:
        m = manifest["categories"][c]
        print(f"[probe]   {c:16s} {m['n']:3d} "
              f"(pc {m['par_coeur']} / rf {m['reformule']})")


if __name__ == "__main__":
    main()
