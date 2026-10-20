"""Pilote QA : corpus d'entraînement GELÉ, format + répétition (déterministe).

But : tester si une recette QA étroite (templates `par_coeur` + répétition
massive) déplace les chiffres de la sonde gelée `data/probe/probe_fr_v1.jsonl`
sur les catégories `capitales` et `unites_sciences`.

Règles
------
- Les FAITS sont extraits de la sonde gelée (source unique de vérité) :
  40 capitales + 20 unités/sciences = 60 faits.
- Seules les formulations **par_coeur** sont utilisées (déclaratives, style
  corpus). Les formulations `reformule` de la sonde sont **interdites** :
  elles doivent rester jamais-vues (c'est le test de généralisation). Le
  générateur refuse bruyamment s'il en détecte une, ou un « ? ».
- Répétition : chaque fait produit `N_VARIANTS` paragraphes distincts
  (rotations/ordres des mêmes phrases), chacun dupliqué `REPEAT` fois.
  `REPEAT` est fixe et documenté dans le manifeste.
- ~5-10 % de texte FR général (extrait déterministe de tranche_05.jsonl)
  pour éviter l'effondrement et garder l'EOS.
- Cible : ~6,19 M tokens (SPE = 379 steps/époque, soit exactement une
  époque neuve depuis le step 22740 de t08).

Sorties
-------
  data/qa/qa_fr_pilot_v1.jsonl            (régénérable, ignoré git)
  data/qa/qa_fr_pilot_v1.manifest.json    (hash SHA256 + comptages, committé)

Usage : .venv/bin/python tools/make_qa_corpus.py [--repeat 19]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
PROBE = os.path.join(ROOT, "data", "probe", "probe_fr_v1.jsonl")
GENERAL = "/Volumes/Lexar/clean-corpus/dedup/tranche_05.jsonl"
OUT_DIR = os.path.join(ROOT, "data", "qa")
OUT_JSONL = os.path.join(OUT_DIR, "qa_fr_pilot_v1.jsonl")
OUT_MANIFEST = os.path.join(OUT_DIR, "qa_fr_pilot_v1.manifest.json")

VERSION = "qa_fr_pilot_v1"
REPEAT = 19          # facteur de répétition (fixe, documenté) — dans 10-20
N_VARIANTS = 40      # paragraphes distincts par fait
TARGET_TOKENS = 6_190_000
SPE_TARGET = 379     # 379 * 60 = 22740 (step t08) -> exactement 1 époque
GENERAL_SEED = 0
GENERAL_DOC_MAX_TOK = 4000
PARENT = "HuggingFaceTB/SmolLM-135M"
VOCAB = os.path.join(ROOT, "runs", "2026-10-08-tranche00",
                     "tranche00.pt.vocab.json")

# --- v2 (pilote QA #2, recette rééquilibrée) -------------------------------
# Dose : ~500 faits × répétition ×4 × ~20 variantes, ~35 % de FR général,
# cible SPE=413 (6,77 M tokens) -> une époque neuve depuis le step 22740 de
# t08 : floor(22740/413)+1 = 56 -> 56*413 = 23128 (restants = 388 ≈ 0,94 SPE).
VERSION_V2 = "qa_fr_pilot_v2"
OUT_JSONL_V2 = os.path.join(OUT_DIR, "qa_fr_pilot_v2.jsonl")
OUT_MANIFEST_V2 = os.path.join(OUT_DIR, "qa_fr_pilot_v2.manifest.json")
REPEAT_V2 = 4                  # répétition (3-5 demandé)
N_VARIANTS_V2 = 13             # paragraphes distincts par fait (534×13×4 tient
                               # dans le budget QA -> 4 répétitions pleines)
SPE_TARGET_V2 = 413            # 413 * 60 = 24780 steps ; une époque depuis 22740
GENERAL_RATIO_V2 = 0.35        # 30-40 % de FR général
GENERAL_LIMIT_V2 = 6000        # docs général candidats (troncature au budget)
GENERAL_DOC_MAX_TOK_V2 = 3000  # plafond par doc général (diversité accrue)

# Catégories couvertes par la v2 (les 4 périmètres de la sonde).
CATEGORIES_V2 = ("capitales", "dates", "definitions", "unites_sciences")

# --- v3 (pilote QA #3, budget x3 + corpus naturel) --------------------------
# Dose : 534 faits x 40 variantes x 10 répétitions, ~22 % de FR général,
# cible SPE=1516 (24,85 M tokens). 1516 x 15 = 22740 (step t08) -> reprise
# à E_CUMUL=16 = exactement une époque neuve (restants = 1516 steps).
VERSION_V3 = "qa_fr_pilot_v3"
OUT_JSONL_V3 = os.path.join(OUT_DIR, "qa_fr_pilot_v3.jsonl")
OUT_MANIFEST_V3 = os.path.join(OUT_DIR, "qa_fr_pilot_v3.manifest.json")
REPEAT_V3 = 10                 # répétition (au lieu de x4 / x19)
N_VARIANTS_V3 = 40             # variantes syntaxiques par fait (~40 demandées)
SPE_TARGET_V3 = 1516           # 1516 * 16384 = 24 846 336 tokens
GENERAL_RATIO_V3 = 0.22        # ~22 % de FR général (au lieu de 35 % / 7 %)
GENERAL_LIMIT_V3 = 12000       # docs général candidats (fichier : ~10 074)
GENERAL_DOC_MAX_TOK_V3 = 3000  # plafond par doc général
# Phrases réelles du corpus FR : récupération déterministe par sous-chaîne
# (frontières de tokens), plafonnée par fait.
REAL_MAX_V3 = 6                # phrases réelles max par fait
REAL_MIN_CHARS_V3 = 30         # longueur minimale d'une phrase retenue
REAL_MAX_CHARS_V3 = 240        # longueur maximale (évite les pavés)
KS_V3 = (6, 7, 8)              # (défaut générique ; la v3 utilise K_PATTERN_V3)
# Longueur (phrases/variante) par catégorie : cible ~90 tokens/doc, pour que
# le QA occupe ~78 % des 24,85 M tokens (≈22 % de FR général).
K_PATTERN_V3 = {
    "capitales": (5,) * 17 + (6,) * 23,
    "dates": (4,) * 38 + (5,) * 2,
    "definitions": (3,) * 20 + (4,) * 20,
    "unites_sciences": (4,) * 26 + (5,) * 14,
}

# Garde anti-fuite : par défaut, les 2 catégories de la v1 (comportement gelé) ;
# la v2 élargit aux 4 catégories de la sonde.
GUARD_CATS_V1 = ("capitales", "unites_sciences")
GUARD_CATS_ALL = CATEGORIES_V2


# ---------------------------------------------------------------------------
# Faits : extraits de la sonde (aucune donnée inventée ici).
# ---------------------------------------------------------------------------

def load_probe_facts(path: str = PROBE) -> tuple[list[dict], list[dict]]:
    """Retourne (capitales, unites) depuis les items `par_coeur` de la sonde."""
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    caps, unis = [], []
    for it in items:
        if it["slice"] != "par_coeur":
            continue
        if it["categorie"] == "capitales":
            de = it["note"].replace("capitale officielle ", "").strip()
            art = _article_from_de(de)
            caps.append({
                "id": it["id"],
                "de": de,
                "pays": art,
                "cap": it["reponses_acceptees"][0],
                "pluriel": art.startswith("Les"),
            })
        elif it["categorie"] == "unites_sciences":
            unis.append({
                "id": it["id"],
                "prompt": it["prompt"],
                "answers": list(it["reponses_acceptees"]),
            })
    return caps, unis


def _article_from_de(de: str) -> str:
    """« de la France » -> « La France » ; « du Portugal » -> « Le Portugal »."""
    if de.startswith("de la "):
        return "La " + de[len("de la "):]
    if de.startswith("de l'"):
        return "L'" + de[len("de l'"):]
    if de.startswith("du "):
        return "Le " + de[len("du "):]
    if de.startswith("des "):
        return "Les " + de[len("des "):]
    raise ValueError(f"forme « de » non reconnue : {de!r}")


def de_from_article(pays: str) -> str:
    """Inverse de `_article_from_de` : « La France » -> « de la France ».

    Gère aussi les noms nus (« Cuba », « Israël », « Oman »).
    """
    if pays.startswith("La "):
        return "de la " + pays[len("La "):]
    if pays.startswith("L'"):
        return "de l'" + pays[len("L'"):]
    if pays.startswith("Le "):
        return "du " + pays[len("Le "):]
    if pays.startswith("Les "):
        return "des " + pays[len("Les "):]
    if pays[:1].lower() in "aeiouâàäéèêëîïôöûü":
        return "d'" + pays
    return "de " + pays


# ---------------------------------------------------------------------------
# Formulations par_coeur (déclaratives uniquement, jamais de « ? »).
# ---------------------------------------------------------------------------

FRAMES_CAPITALES = [
    "La capitale {de} est {cap}.",
    "{cap} est la capitale {de}.",
    "La capitale {de}, c'est {cap}.",
    "{pays} {avoir} pour capitale {cap}.",
    "Le chef-lieu {de} est {cap}.",
    "{pays} : capitale {cap}.",
    "La ville de {cap} est la capitale {de}.",
    "{pays} {etre} un pays dont la capitale est {cap}.",
    "On appelle {cap} la capitale {de}.",
    "La capitale {de} porte le nom de {cap}.",
    "{pays} {compter} {cap} comme capitale.",
    "Sur le plan administratif, la capitale {de} est {cap}.",
]

# Phrases des faits « unites_sciences » (10 formulations par fait, toutes
# déclaratives et vraies ; clés = id de la sonde).
UNIT_SENTENCES = {
    "uni-01-pc": [
        "L'eau bout à 100 degrés Celsius.",
        "L'eau bout à 100 °C.",
        "Le point d'ébullition de l'eau est de 100 degrés Celsius.",
        "La température d'ébullition de l'eau est de 100 °C.",
        "L'eau bout à une température de 100 degrés.",
        "À 100 degrés Celsius, l'eau se met à bouillir.",
        "Le point d'ébullition de l'eau est de 100 degrés.",
        "L'eau atteint son point d'ébullition à 100 °C.",
        "L'eau bout à 100 degrés à pression atmosphérique normale.",
        "La température d'ébullition de l'eau vaut 100 degrés Celsius.",
    ],
    "uni-02-pc": [
        "L'eau gèle à 0 degré Celsius.",
        "L'eau gèle à 0 °C.",
        "Le point de congélation de l'eau est de 0 degré Celsius.",
        "La température de congélation de l'eau est de 0 °C.",
        "L'eau gèle à une température de 0 degré.",
        "À 0 degré Celsius, l'eau se met à geler.",
        "Le point de fusion de l'eau est de 0 degré.",
        "L'eau atteint son point de congélation à 0 °C.",
        "L'eau gèle à 0 degré à pression atmosphérique normale.",
        "La température de fusion de l'eau vaut 0 degré Celsius.",
    ],
    "uni-03-pc": [
        "Un kilomètre vaut 1000 mètres.",
        "Un kilomètre vaut 1000 m.",
        "Un kilomètre mesure 1000 mètres.",
        "Un kilomètre correspond à 1000 mètres.",
        "Un kilomètre représente 1000 mètres.",
        "Il y a 1000 mètres dans un kilomètre.",
        "Un kilomètre est égal à 1000 mètres.",
        "La longueur d'un kilomètre est de 1000 mètres.",
        "Un kilomètre contient 1000 mètres.",
        "Un kilomètre vaut mille mètres.",
    ],
    "uni-04-pc": [
        "Un mètre contient 100 centimètres.",
        "Un mètre vaut 100 centimètres.",
        "Un mètre mesure 100 centimètres.",
        "Un mètre correspond à 100 centimètres.",
        "Un mètre représente 100 centimètres.",
        "Il y a 100 centimètres dans un mètre.",
        "Un mètre est égal à 100 centimètres.",
        "La longueur d'un mètre est de 100 centimètres.",
        "Un mètre se divise en 100 centimètres.",
        "Un mètre vaut cent centimètres.",
    ],
    "uni-05-pc": [
        "Une heure dure 60 minutes.",
        "Une heure vaut 60 minutes.",
        "Une heure compte 60 minutes.",
        "Une heure correspond à 60 minutes.",
        "Une heure représente 60 minutes.",
        "Il y a 60 minutes dans une heure.",
        "Une heure est égale à 60 minutes.",
        "La durée d'une heure est de 60 minutes.",
        "Une heure se divise en 60 minutes.",
        "Une heure dure soixante minutes.",
    ],
    "uni-06-pc": [
        "Un kilogramme vaut 1000 grammes.",
        "Un kilogramme vaut 1000 g.",
        "Un kilogramme mesure 1000 grammes.",
        "Un kilogramme correspond à 1000 grammes.",
        "Un kilogramme représente 1000 grammes.",
        "Il y a 1000 grammes dans un kilogramme.",
        "Un kilogramme est égal à 1000 grammes.",
        "La masse d'un kilogramme est de 1000 grammes.",
        "Un kilogramme contient 1000 grammes.",
        "Un kilogramme vaut mille grammes.",
    ],
    "uni-07-pc": [
        "Un litre vaut 1000 millilitres.",
        "Un litre vaut 1000 ml.",
        "Un litre mesure 1000 millilitres.",
        "Un litre correspond à 1000 millilitres.",
        "Un litre représente 1000 millilitres.",
        "Il y a 1000 millilitres dans un litre.",
        "Un litre est égal à 1000 millilitres.",
        "Le volume d'un litre est de 1000 millilitres.",
        "Un litre contient 1000 millilitres.",
        "Un litre vaut mille millilitres.",
    ],
    "uni-08-pc": [
        "La vitesse de la lumière est de 300000 kilomètres par seconde.",
        "La vitesse de la lumière vaut 300000 km/s.",
        "La lumière se déplace à 300000 kilomètres par seconde.",
        "La vitesse de la lumière dans le vide est de 299792 kilomètres par seconde.",
        "La vitesse de la lumière est d'environ 300000 kilomètres par seconde.",
        "La lumière voyage à 300000 km/s dans le vide.",
        "La vitesse de la lumière est de 299792 km/s.",
        "La célérité de la lumière vaut 300000 kilomètres par seconde.",
        "Dans le vide, la lumière se propage à 300000 km/s.",
        "La vitesse de la lumière est proche de 300000 kilomètres par seconde.",
    ],
    "uni-09-pc": [
        "Le corps humain compte 206 os.",
        "Le squelette humain compte 206 os.",
        "Le corps humain possède 206 os.",
        "Un être humain compte 206 os.",
        "Le nombre d'os du corps humain est de 206.",
        "Le squelette d'un adulte compte 206 os.",
        "Le corps humain est composé de 206 os.",
        "L'être humain possède 206 os.",
        "Le corps humain comporte 206 os.",
        "Le nombre d'os du squelette humain vaut 206.",
    ],
    "uni-10-pc": [
        "La température du corps humain est de 37 degrés Celsius.",
        "La température normale du corps humain est de 37 °C.",
        "Le corps humain a une température de 37 degrés.",
        "La température corporelle normale est de 37 °C.",
        "La température du corps humain vaut 37 degrés Celsius.",
        "Chez l'être humain, la température corporelle est de 37 degrés.",
        "La température interne du corps humain est de 37 °C.",
        "La température normale du corps est de 37 degrés Celsius.",
        "Le corps humain maintient une température de 37 degrés.",
        "La température du corps humain est d'environ 37 degrés.",
    ],
    "uni-11-pc": [
        "Le système solaire compte 8 planètes.",
        "Le système solaire compte huit planètes.",
        "Le système solaire possède 8 planètes.",
        "Le système solaire comporte 8 planètes.",
        "Le nombre de planètes du système solaire est de 8.",
        "Huit planètes composent le système solaire.",
        "Le système solaire est composé de 8 planètes.",
        "Le système solaire contient 8 planètes.",
        "Le système solaire réunit 8 planètes.",
        "Le nombre de planètes du système solaire vaut 8.",
    ],
    "uni-12-pc": [
        "Une journée compte 86400 secondes.",
        "Une journée dure 86400 secondes.",
        "Une journée vaut 86400 secondes.",
        "Une journée correspond à 86400 secondes.",
        "Il y a 86400 secondes dans une journée.",
        "Une journée est égale à 86400 secondes.",
        "La durée d'une journée est de 86400 secondes.",
        "Une journée représente 86400 secondes.",
        "Une journée se compose de 86400 secondes.",
        "Une journée contient 86400 secondes.",
    ],
    "uni-13-pc": [
        "La Terre tourne autour du Soleil en 365 jours.",
        "La Terre fait le tour du Soleil en 365 jours.",
        "La Terre orbite autour du Soleil en 365 jours.",
        "Une révolution de la Terre autour du Soleil dure 365 jours.",
        "La Terre accomplit son orbite en 365 jours.",
        "La période orbitale de la Terre est de 365 jours.",
        "La Terre met 365 jours pour faire le tour du Soleil.",
        "Le tour du Soleil par la Terre dure 365 jours.",
        "La Terre parcourt son orbite en 365 jours.",
        "La durée d'une année terrestre est de 365 jours.",
    ],
    "uni-14-pc": [
        "Le symbole chimique de l'or est Au.",
        "L'or a pour symbole chimique Au.",
        "Le symbole de l'or est Au.",
        "L'élément or porte le symbole Au.",
        "En chimie, l'or est noté Au.",
        "Le symbole chimique de l'élément or est Au.",
        "L'or correspond au symbole Au.",
        "Le symbole de l'élément or est Au.",
        "L'or est représenté par le symbole Au.",
        "Le symbole chimique de l'or s'écrit Au.",
    ],
    "uni-15-pc": [
        "La molécule d'eau est H2O.",
        "La formule chimique de l'eau est H2O.",
        "L'eau a pour formule H2O.",
        "La molécule d'eau a pour formule H2O.",
        "L'eau est composée de H2O.",
        "La formule de la molécule d'eau est H2O.",
        "En chimie, l'eau s'écrit H2O.",
        "La molécule d'eau correspond à H2O.",
        "L'eau est représentée par la formule H2O.",
        "La formule chimique de la molécule d'eau est H2O.",
    ],
    "uni-16-pc": [
        "L'accélération de la pesanteur est de 9,8 mètres par seconde au carré.",
        "L'accélération de la pesanteur vaut 9,8 m/s².",
        "La pesanteur terrestre est de 9,8 m/s².",
        "L'accélération due à la pesanteur est de 9,8 mètres par seconde carrée.",
        "Sur Terre, l'accélération de la pesanteur vaut 9,8 m/s².",
        "La constante de pesanteur terrestre est de 9,8 m/s².",
        "L'accélération gravitationnelle terrestre est de 9,8 m/s².",
        "La pesanteur à la surface de la Terre est de 9,8 m/s².",
        "L'accélération de la pesanteur terrestre est d'environ 9,8 m/s².",
        "Le champ de pesanteur terrestre vaut 9,8 m/s².",
    ],
    "uni-17-pc": [
        "Le point d'ébullition de l'azote est de -196 degrés Celsius.",
        "L'azote bout à -196 °C.",
        "Le point d'ébullition de l'azote vaut -196 °C.",
        "L'azote liquide bout à -196 degrés Celsius.",
        "La température d'ébullition de l'azote est de -196 °C.",
        "L'azote passe à l'état gazeux à -196 degrés.",
        "Le point d'ébullition de l'azote liquide est de -196 °C.",
        "À -196 degrés Celsius, l'azote se met à bouillir.",
        "L'azote bout à une température de -196 °C.",
        "La température d'ébullition de l'azote vaut -196 degrés Celsius.",
    ],
    "uni-18-pc": [
        "Le nombre de chromosomes humains est de 46.",
        "Une cellule humaine compte 46 chromosomes.",
        "Le génome humain compte 46 chromosomes.",
        "L'être humain possède 46 chromosomes.",
        "Une cellule humaine possède 46 chromosomes.",
        "Le nombre de chromosomes d'une cellule humaine est de 46.",
        "Les cellules humaines contiennent 46 chromosomes.",
        "Le patrimoine génétique humain compte 46 chromosomes.",
        "Une cellule humaine renferme 46 chromosomes.",
        "Le nombre de chromosomes chez l'être humain vaut 46.",
    ],
    "uni-19-pc": [
        "Une année compte 365 jours.",
        "Une année non bissextile compte 365 jours.",
        "Une année dure 365 jours.",
        "Une année vaut 365 jours.",
        "Une année correspond à 365 jours.",
        "Il y a 365 jours dans une année.",
        "Une année est égale à 365 jours.",
        "La durée d'une année est de 365 jours.",
        "Une année civile compte 365 jours.",
        "Une année représente 365 jours.",
    ],
    "uni-20-pc": [
        "La température d'ébullition de l'eau est de 212 degrés Fahrenheit.",
        "L'eau bout à 212 degrés Fahrenheit.",
        "Le point d'ébullition de l'eau est de 212 °F.",
        "L'eau bout à 212 °F.",
        "La température d'ébullition de l'eau vaut 212 degrés Fahrenheit.",
        "À 212 degrés Fahrenheit, l'eau se met à bouillir.",
        "Le point d'ébullition de l'eau vaut 212 °F.",
        "L'eau atteint son point d'ébullition à 212 °F.",
        "La température d'ébullition de l'eau est de 212 degrés Fahrenheit à pression normale.",
        "L'eau bout à une température de 212 degrés Fahrenheit.",
    ],
}


def _sentences_capitales(fact: dict) -> list[str]:
    out = []
    for fr in FRAMES_CAPITALES:
        # « Les Pays-Bas est un pays dont la capitale est » : le prompt
        # par_coeur de la sonde (cap-33) utilise « est » ; on garde la
        # formulation telle quelle pour que le préfixe reste mémorisable.
        out.append(fr.format(
            de=fact["de"], cap=fact["cap"], pays=fact["pays"],
            avoir="ont" if fact["pluriel"] else "a",
            etre="est",
            compter="comptent" if fact["pluriel"] else "compte",
        ))
    return out


def _paragraphs(sentences: list[str], n_variants: int = N_VARIANTS) -> list[str]:
    """n_variants paragraphes distincts : rotations x ordres + versions courtes."""
    n = len(sentences)
    seen, variants = set(), []

    def add(text: str) -> None:
        if text not in seen:
            seen.add(text)
            variants.append(text)

    modes = (
        lambda r: r,
        lambda r: list(reversed(r)),
        lambda r: r[0::2] + r[1::2],
        lambda r: r[1::2] + r[0::2],
    )
    # Passes longues (toutes les phrases), puis passes courtes (6 phrases).
    for k in (n, 6):
        for mode in modes:
            for off in range(n):
                rot = sentences[off:] + sentences[:off]
                add(" ".join(mode(rot)[:k]))
                if len(variants) >= n_variants:
                    break
            if len(variants) >= n_variants:
                break
        if len(variants) >= n_variants:
            break
    assert len(variants) == n_variants, (len(variants), n)
    return variants


def build_facts() -> list[dict]:
    """60 faits -> liste de paragraphes (un fait = N_VARIANTS paragraphes)."""
    caps, unis = load_probe_facts()
    facts = []
    for f in caps:
        sents = _sentences_capitales(f)
        assert len(sents) >= 10, f
        facts.append({"id": f["id"], "categorie": "capitales",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents)})
    for f in unis:
        sents = UNIT_SENTENCES.get(f["id"])
        assert sents, f"phrases manquantes pour {f['id']}"
        facts.append({"id": f["id"], "categorie": "unites_sciences",
                      "sentences": list(sents),
                      "paragraphs": _paragraphs(list(sents))})
    assert len(facts) == 60, len(facts)
    return facts


# ---------------------------------------------------------------------------
# v2 : formulations par_coeur des 4 catégories (déclaratives, sans « ? »).
# ---------------------------------------------------------------------------

FRAMES_DATES = [
    "{sujet} a eu lieu en {annee}.",
    "{sujet} date de {annee}.",
    "{Sujet} a eu lieu en {annee}.",
    "{Sujet} date de {annee}.",
    "{Sujet} : {annee}.",
    "{sujet} : {annee}.",
    "En {annee} a eu lieu {sujet}.",
    "L'année {annee} est celle de {sujet}.",
]

# Le complément est un groupe nominal : chaque trame le fait suivre d'un
# verbe/connecteur neutre (aucune contraction « à le »).
FRAMES_DEFINITIONS = [
    "{terme_art} est {complement}.",
    "Le mot « {nu} » désigne {complement}.",
    "{terme_art} : {complement}.",
    "On définit {terme_art} comme {complement}.",
    "{terme_art}, c'est {complement}.",
    "Par {nu}, on entend {complement}.",
    "Le mot « {nu} » évoque {complement}.",
    "Dans {terme_art}, il y a {complement}.",
    "Le mot « {nu} », c'est {complement}.",
]

# Unités/sciences : {sujet} + {verbe} + {reponse} + {unite} (unité optionnelle).
FRAMES_UNITS = [
    "{sujet} {verbe} {val}.",
    "{sujet} {verbe} {val}, c'est une valeur de référence.",
    "{sujet} : {val}.",
    "Rappel : {sujet_min} {verbe} {val}.",
    "On retient que {sujet_min} {verbe} {val}.",
    "{sujet} {verbe} {val}, une constante de référence.",
]


def _cap1(s: str) -> str:
    return s[:1].upper() + s[1:]


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:]


def _sentences_dates(sujet: str, annee: str) -> list[str]:
    return [fr.format(sujet=sujet, Sujet=_cap1(sujet), annee=annee)
            for fr in FRAMES_DATES]


def _sentences_definitions(terme_art: str, nu: str,
                           complement: str) -> list[str]:
    return [fr.format(terme_art=terme_art, nu=nu, complement=complement)
            for fr in FRAMES_DEFINITIONS]


def _sentences_units(sujet: str, verbe: str, a: str, unite: str) -> list[str]:
    val = (a + " " + unite).strip() if unite else a
    return [fr.format(sujet=sujet, sujet_min=_lower1(sujet), verbe=verbe,
                      val=val) for fr in FRAMES_UNITS]


def _probe_pc_items(path: str = PROBE) -> list[dict]:
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [it for it in items if it["slice"] == "par_coeur"]


def _date_sujet(prompt: str) -> str:
    for suf in (" a eu lieu en", " date de"):
        if prompt.endswith(suf):
            return prompt[: -len(suf)]
    raise ValueError(f"prompt de date non reconnu : {prompt!r}")


def build_facts_v2() -> list[dict]:
    """~500 faits (les 100 pc de la sonde + extras) -> N_VARIANTS_V2 paragraphes.

    Catégories : capitales, dates, definitions, unites_sciences.
    """
    import tools.qa_facts_v2 as F2

    pc = _probe_pc_items()
    facts: list[dict] = []
    seen_pays: set[str] = set()

    # --- capitales : les 40 de la sonde, puis les extras ---------------------
    caps, unis = load_probe_facts()
    for f in caps:
        seen_pays.add(f["pays"])
        sents = _sentences_capitales(f)
        assert len(sents) >= 10, f
        facts.append({"id": f["id"], "categorie": "capitales",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})
    for i, (pays, cap) in enumerate(F2.CAPITALES_EXTRA, start=1):
        if pays in seen_pays:
            continue
        seen_pays.add(pays)
        f = {"id": f"cap-x{i:03d}", "de": de_from_article(pays), "pays": pays,
             "cap": cap, "pluriel": pays.startswith("Les")}
        sents = _sentences_capitales(f)
        facts.append({"id": f["id"], "categorie": "capitales",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})

    # --- dates : les 20 de la sonde, puis les extras -------------------------
    for it in pc:
        if it["categorie"] != "dates":
            continue
        sujet = _date_sujet(it["prompt"])
        annee = it["reponses_acceptees"][0]
        sents = _sentences_dates(sujet, annee)
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]
        facts.append({"id": it["id"], "categorie": "dates",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})
    for i, (sujet, annee) in enumerate(F2.DATES_EXTRA, start=1):
        sents = _sentences_dates(sujet, annee)
        facts.append({"id": f"dat-x{i:03d}", "categorie": "dates",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})

    # --- definitions : les 20 de la sonde, puis les extras -------------------
    for it in pc:
        if it["categorie"] != "definitions":
            continue
        terme_art, nu, _kw, complement = F2.DEFINITIONS_PROBE[it["id"]]
        sents = _sentences_definitions(terme_art, nu, complement)
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]
        facts.append({"id": it["id"], "categorie": "definitions",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})
    for i, (terme_art, nu, _kw, complement) in enumerate(
            F2.DEFINITIONS_EXTRA, start=1):
        sents = _sentences_definitions(terme_art, nu, complement)
        facts.append({"id": f"def-x{i:03d}", "categorie": "definitions",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})

    # --- unites_sciences : les 20 de la sonde, puis les extras ---------------
    for f in unis:
        sents = UNIT_SENTENCES.get(f["id"])
        assert sents, f"phrases manquantes pour {f['id']}"
        items = list(sents)
        assert any(s.startswith(f["prompt"]) for s in items), f["id"]
        facts.append({"id": f["id"], "categorie": "unites_sciences",
                      "sentences": items,
                      "paragraphs": _paragraphs(items, N_VARIANTS_V2)})
    for i, (sujet, verbe, a, unite, _kw) in enumerate(F2.UNITS_EXTRA, start=1):
        sents = _sentences_units(sujet, verbe, a, unite)
        facts.append({"id": f"uni-x{i:03d}", "categorie": "unites_sciences",
                      "sentences": sents,
                      "paragraphs": _paragraphs(sents, N_VARIANTS_V2)})

    for f in facts:
        assert len(f["paragraphs"]) == N_VARIANTS_V2, f["id"]
        assert len(set(f["paragraphs"])) == N_VARIANTS_V2, f["id"]
    return facts


# ---------------------------------------------------------------------------
# v3 : corpus naturel (40 variantes/fait + phrases réelles de tranche_05).
# ---------------------------------------------------------------------------

# Trames élargies (déclaratives, sans « ? », vérité = fait de la sonde).
FRAMES_CAPITALES_V3 = [
    "La capitale {de} est {cap}.",
    "{cap} est la capitale {de}.",
    "La capitale {de}, c'est {cap}.",
    "{pays} {avoir} pour capitale {cap}.",
    "Le chef-lieu {de} est {cap}.",
    "{pays} : capitale {cap}.",
    "La ville de {cap} est la capitale {de}.",
    "{pays} {etre} un pays dont la capitale est {cap}.",
    "On appelle {cap} la capitale {de}.",
    "La capitale {de} porte le nom de {cap}.",
    "{pays} {compter} {cap} comme capitale.",
    "Sur le plan administratif, la capitale {de} est {cap}.",
    "{en_pays}, la capitale est {cap}.",
    "Capitale {de} : {cap}.",
    "Quand on parle {de}, on pense à {cap}, sa capitale.",
    "C'est {cap} qui est la capitale {de}.",
    "Administrativement, {cap} est la capitale {de}.",
    "{pays} {avoir} {cap} comme capitale.",
]

FRAMES_DATES_V3 = [
    "{sujet} {a_eu} en {annee}.",
    "{sujet} {date} de {annee}.",
    "On retient {annee} pour {sujet}.",
    "La date retenue est {annee} pour {sujet}.",
    "{Sujet} : {annee}.",
    "Date : {annee} pour {sujet}.",
    "En {annee} {a_eu} {sujet}.",
    "L'année {annee} est celle de {sujet}.",
    "On situe {sujet} en {annee}.",
    "{Sujet} {remonte} à {annee}.",
    "{Sujet} {situe} en {annee}.",
    "C'est en {annee} {qu_a_eu} {sujet}.",
    "La date de {sujet} est {annee}.",
    "{Sujet} {correspond} à l'année {annee}.",
    "En {annee}, {sujet}.",
    "Il faut rattacher {sujet} à {annee}.",
]

FRAMES_DEFINITIONS_V3 = [
    "{terme_art} est {complement}.",
    "Le mot « {nu} » désigne {complement}.",
    "{terme_art} : {complement}.",
    "On définit {terme_art} comme {complement}.",
    "{terme_art}, c'est {complement}.",
    "Par {terme_art}, on entend {complement}.",
    "Le mot « {nu} » évoque {complement}.",
    "On associe {terme_art} à {complement}.",
    "Le mot « {nu} », c'est {complement}.",
    "{terme_art} renvoie à {complement}.",
    "{terme_art} se définit comme {complement}.",
    "Par définition, {terme_art} est {complement}.",
    "Le terme « {nu} » signifie {complement}.",
    "{terme_art} correspond à {complement}.",
    "{terme_art} représente {complement}.",
    "On retient {que_art} est {complement}.",
]

FRAMES_UNITS_V3 = [
    "{sujet} {verbe} {val}.",
    "{sujet} {verbe} {val}, c'est une valeur de référence.",
    "{sujet} : {val}.",
    "Rappel : {sujet_min} {verbe} {val}.",
    "On retient {que_sujet} {verbe} {val}.",
    "{sujet} {verbe} {val}, une constante de référence.",
    "En valeur, {sujet_min} {verbe} {val}.",
    "Le repère à connaître : {sujet_min} {verbe} {val}.",
    "{sujet} correspond à {val}.",
    "On sait {que_sujet} {verbe} {val}.",
    "À retenir : {sujet_min} {verbe} {val}.",
    "La valeur à retenir est {val} pour {sujet_min}.",
    "{sujet} : on retient {val}.",
    "Retenons {que_sujet} {verbe} {val}.",
]

# Triplets (sujet, verbe, valeur) des 20 unités de la sonde : permet
# d'appliquer les trames génériques aux faits de la sonde (au lieu des seules
# phrases UNIT_SENTENCES), tout en gardant le préfixe par_coeur.
UNIT_TRIPLES = {
    "uni-01-pc": ("Le point d'ébullition de l'eau", "est de",
                  "100 degrés Celsius"),
    "uni-02-pc": ("Le point de congélation de l'eau", "est de",
                  "0 degré Celsius"),
    "uni-03-pc": ("Un kilomètre", "vaut", "1000 mètres"),
    "uni-04-pc": ("Un mètre", "vaut", "100 centimètres"),
    "uni-05-pc": ("Une heure", "dure", "60 minutes"),
    "uni-06-pc": ("Un kilogramme", "vaut", "1000 grammes"),
    "uni-07-pc": ("Un litre", "vaut", "1000 millilitres"),
    "uni-08-pc": ("La vitesse de la lumière", "est de",
                  "300000 kilomètres par seconde"),
    "uni-09-pc": ("Le corps humain", "compte", "206 os"),
    "uni-10-pc": ("La température du corps humain", "est de",
                  "37 degrés Celsius"),
    "uni-11-pc": ("Le système solaire", "compte", "8 planètes"),
    "uni-12-pc": ("Une journée", "compte", "86400 secondes"),
    "uni-13-pc": ("La période orbitale de la Terre", "est de", "365 jours"),
    "uni-14-pc": ("Le symbole chimique de l'or", "est", "Au"),
    "uni-15-pc": ("La molécule d'eau", "est", "H2O"),
    "uni-16-pc": ("L'accélération de la pesanteur", "est de",
                  "9,8 mètres par seconde au carré"),
    "uni-17-pc": ("Le point d'ébullition de l'azote", "est de",
                  "-196 degrés Celsius"),
    "uni-18-pc": ("Une cellule humaine", "compte", "46 chromosomes"),
    "uni-19-pc": ("Une année", "compte", "365 jours"),
    "uni-20-pc": ("Le point d'ébullition de l'eau", "est de",
                  "212 degrés Fahrenheit"),
}

# Entité + réponse (ancre de récupération des phrases réelles) des 20 unités.
UNIT_ENTITY = {
    "uni-01-pc": ("eau", "100"), "uni-02-pc": ("eau", "0"),
    "uni-03-pc": ("kilomètre", "1000"), "uni-04-pc": ("mètre", "100"),
    "uni-05-pc": ("heure", "60"), "uni-06-pc": ("kilogramme", "1000"),
    "uni-07-pc": ("litre", "1000"), "uni-08-pc": ("lumière", "300000"),
    "uni-09-pc": ("corps humain", "206"), "uni-10-pc": ("corps humain", "37"),
    "uni-11-pc": ("système solaire", "8"), "uni-12-pc": ("journée", "86400"),
    "uni-13-pc": ("Terre", "365"), "uni-14-pc": ("or", "Au"),
    "uni-15-pc": ("eau", "H2O"), "uni-16-pc": ("pesanteur", "9,8"),
    "uni-17-pc": ("azote", "196"), "uni-18-pc": ("cellule humaine", "46"),
    "uni-19-pc": ("année", "365"), "uni-20-pc": ("eau", "212"),
}


def _en_from_article(pays: str) -> str:
    """« La France » -> « en France » ; « Le Portugal » -> « au Portugal »."""
    if pays.startswith("La "):
        return "en " + pays[3:]
    if pays.startswith("L'"):
        return "en " + pays[2:]
    if pays.startswith("Le "):
        return "au " + pays[3:]
    if pays.startswith("Les "):
        return "aux " + pays[4:]
    return "en " + pays


def _pays_nu(pays: str) -> str:
    """« L'Afghanistan » -> « Afghanistan » (nom sans article)."""
    for pre in ("La ", "Le ", "Les ", "L'"):
        if pays.startswith(pre):
            return pays[len(pre):]
    return pays


_VOYELLES = "aeiouyàâäéèêëîïôöûüù"


def _elide_que(s: str) -> str:
    """« que la ... » / « qu'une ... » (élision devant voyelle)."""
    return ("qu'" if s[:1].lower() in _VOYELLES else "que ") + s


def _sentences_capitales_v3(fact: dict) -> list[str]:
    out = []
    for fr in FRAMES_CAPITALES_V3:
        out.append(fr.format(
            de=fact["de"], cap=fact["cap"], pays=fact["pays"],
            en_pays=_en_from_article(fact["pays"]),
            avoir="ont" if fact["pluriel"] else "a",
            etre="est",
            compter="comptent" if fact["pluriel"] else "compte",
        ))
    return out


def _sentences_dates_v3(sujet: str, annee: str) -> list[str]:
    pl = sujet.lower().startswith(("les ", "des "))
    v = {
        "a_eu": "ont eu lieu" if pl else "a eu lieu",
        "date": "datent" if pl else "date",
        "qu_a_eu": "qu'ont eu lieu" if pl else "qu'a eu lieu",
        "remonte": "remontent" if pl else "remonte",
        "situe": "se situent" if pl else "se situe",
        "correspond": "correspondent" if pl else "correspond",
    }
    return [fr.format(sujet=sujet, Sujet=_cap1(sujet), annee=annee, **v)
            for fr in FRAMES_DATES_V3]


def _sentences_definitions_v3(terme_art: str, nu: str,
                              complement: str) -> list[str]:
    return [fr.format(terme_art=terme_art, Terme_art=_cap1(terme_art), nu=nu,
                      complement=complement, que_art=_elide_que(terme_art))
            for fr in FRAMES_DEFINITIONS_V3]


def _sentences_units_v3(sujet: str, verbe: str, a: str, unite: str) -> list[str]:
    val = (a + " " + unite).strip() if unite else a
    return [fr.format(sujet=sujet, sujet_min=_lower1(sujet), verbe=verbe,
                      val=val, que_sujet=_elide_que(_lower1(sujet)))
            for fr in FRAMES_UNITS_V3]


# ---------------------------------------------------------------------------
# Phrases réelles de tranche_05 (déterministe, plafonnée par fait).
# ---------------------------------------------------------------------------

def _has_token(hay_norm: str, needle_norm: str) -> bool:
    """Sous-chaîne avec frontières de tokens (`100` ne matche pas `1100`)."""
    if not needle_norm:
        return False
    return (" " + needle_norm + " ") in (" " + hay_norm + " ")


def _real_sentence_meta() -> list[dict]:
    """Métadonnées (id, catégorie, ancre, réponse) des 534 faits de la v3."""
    import tools.qa_facts_v2 as F2

    pc = _probe_pc_items()
    meta: list[dict] = []

    caps, unis = load_probe_facts()
    for f in caps:
        meta.append({"id": f["id"], "categorie": "capitales",
                     "anchor": f["cap"], "other": _pays_nu(f["pays"])})
    for i, (pays, cap) in enumerate(F2.CAPITALES_EXTRA, start=1):
        meta.append({"id": f"cap-x{i:03d}", "categorie": "capitales",
                     "anchor": cap, "other": _pays_nu(pays)})

    for it in pc:
        if it["categorie"] != "dates":
            continue
        meta.append({"id": it["id"], "categorie": "dates",
                     "anchor": _date_sujet(it["prompt"]),
                     "other": it["reponses_acceptees"][0]})
    for i, (sujet, annee) in enumerate(F2.DATES_EXTRA, start=1):
        meta.append({"id": f"dat-x{i:03d}", "categorie": "dates",
                     "anchor": sujet, "other": annee})

    for it in pc:
        if it["categorie"] != "definitions":
            continue
        _art, nu, kws, _comp = F2.DEFINITIONS_PROBE[it["id"]]
        meta.append({"id": it["id"], "categorie": "definitions",
                     "anchor": nu, "other": kws[0]})
    for i, (_art, nu, kws, _comp) in enumerate(F2.DEFINITIONS_EXTRA, start=1):
        meta.append({"id": f"def-x{i:03d}", "categorie": "definitions",
                     "anchor": nu, "other": kws[0]})

    for f in unis:
        ent, rep = UNIT_ENTITY[f["id"]]
        meta.append({"id": f["id"], "categorie": "unites_sciences",
                     "anchor": ent, "other": rep})
    for i, (sujet, _verbe, _a, _unite, kws) in enumerate(F2.UNITS_EXTRA, start=1):
        meta.append({"id": f"uni-x{i:03d}", "categorie": "unites_sciences",
                     "anchor": sujet, "other": kws[0]})
    return meta


_REAL_CACHE: dict = {}


def retrieve_real_sentences(meta: list[dict], corpus: str = GENERAL,
                            max_per: int = REAL_MAX_V3,
                            smin: int = REAL_MIN_CHARS_V3,
                            smax: int = REAL_MAX_CHARS_V3) -> dict[str, list[str]]:
    """Phrases réelles contenant l'entité ET la réponse, plafonnées par fait.

    Parcours séquentiel de `corpus` (ordre du fichier) -> déterministe.
    Les phrases sont normalisées pour le test, mais renvoyées telles quelles
    (morphologie naturelle préservée). Aucune phrase interrogative.
    """
    key = (corpus, max_per, smin, smax,
           tuple((m["id"], m["categorie"], m["anchor"], m["other"])
                 for m in meta))
    if key in _REAL_CACHE:
        return _REAL_CACHE[key]

    index: dict[str, list[tuple]] = {}
    for m in meta:
        anchor_n = _norm(m["anchor"])
        words = [w for w in re.findall(r"[a-z0-9]+", anchor_n)
                 if len(w) >= 3]
        k = max(words, key=len) if words else anchor_n
        index.setdefault(k, []).append(
            (m["id"], m["categorie"], anchor_n, _norm(m["other"])))
    keyset = set(index)

    found: dict[str, list[str]] = {m["id"]: [] for m in meta}
    n_meta = len(meta)
    full = 0
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            txt = json.loads(line).get("text", "")
            tokens = set(re.findall(r"[a-z0-9]+", _norm(txt)))
            cands = keyset & tokens
            if not cands:
                continue
            for match in re.finditer(r"[^.!?\n]+[.!?]?", txt):
                s = " ".join(match.group(0).split())
                if not (smin <= len(s) <= smax) or "?" in s:
                    continue
                ls = _norm(s)
                for k in sorted(cands):
                    for fid, cat, a, b in index[k]:
                        if len(found[fid]) >= max_per:
                            continue
                        if not _has_token(ls, a) or not _has_token(ls, b):
                            continue
                        # Capitales : exiger le mot « capitale » (pertinence).
                        if cat == "capitales" and "capitale" not in ls:
                            continue
                        if s not in found[fid]:
                            found[fid].append(s)
                            if len(found[fid]) == max_per:
                                full += 1
            if full == n_meta:
                break
    _REAL_CACHE[key] = found
    return found


def _paragraphs_v3(sentences: list[str], n_variants: int = N_VARIANTS_V3,
                   ks: tuple | None = None) -> list[str]:
    """n_variants paragraphes déterministes (rotations, longueurs variées).

    `ks[i]` = nombre de phrases du paragraphe i ; les ordres varient par
    rotation et par pas (1, 2, 3, 5, 7, 11) pour éviter les doublons.
    """
    n = len(sentences)
    if ks is None:
        ks = (5,) * 16 + (6,) * 24
    if len(ks) < n_variants:
        ks = tuple(ks[i % len(ks)] for i in range(n_variants))
    seen, out = set(), []
    for i in range(n_variants):
        k = min(ks[i], n)
        found = False
        for stride in (1, 2, 3, 5, 7, 11, 13, 17):
            for delta in range(n):
                off = (i * 7 + delta) % n
                text = " ".join(sentences[(off + j * stride) % n]
                                for j in range(k))
                if text not in seen:
                    seen.add(text)
                    out.append(text)
                    found = True
                    break
            if found:
                break
        if not found:  # pragma: no cover - garde-fou
            raise AssertionError(f"variante impossible ({i}, n={n})")
    assert len(out) == n_variants, (len(out), n)
    return out


def build_facts_v3() -> list[dict]:
    """534 faits -> 40 variantes (trames élargies + phrases réelles)."""
    import tools.qa_facts_v2 as F2

    meta = _real_sentence_meta()
    assert len(meta) == 534, len(meta)
    real = retrieve_real_sentences(meta)
    facts: list[dict] = []
    seen_pays: set[str] = set()

    def attach(fact: dict, sents: list[str]) -> None:
        rs = real.get(fact["id"], [])
        for s in rs:
            assert "?" not in s, s[:80]
        sents = list(sents) + list(rs)
        assert len(sents) >= 14, (fact["id"], len(sents))
        fact["sentences"] = sents
        fact["real_sentences"] = rs
        fact["paragraphs"] = _paragraphs_v3(
            sents, ks=K_PATTERN_V3[fact["categorie"]])
        facts.append(fact)

    caps, unis = load_probe_facts()
    for f in caps:
        seen_pays.add(f["pays"])
        attach({"id": f["id"], "categorie": "capitales", "de": f["de"],
                "pays": f["pays"], "cap": f["cap"], "pluriel": f["pluriel"]},
               _sentences_capitales_v3(f))
    for i, (pays, cap) in enumerate(F2.CAPITALES_EXTRA, start=1):
        if pays in seen_pays:
            continue
        seen_pays.add(pays)
        f = {"id": f"cap-x{i:03d}", "de": de_from_article(pays), "pays": pays,
             "cap": cap, "pluriel": pays.startswith("Les")}
        attach({"id": f["id"], "categorie": "capitales", "de": f["de"],
                "pays": pays, "cap": cap, "pluriel": f["pluriel"]},
               _sentences_capitales_v3(f))

    for it in _probe_pc_items():
        if it["categorie"] != "dates":
            continue
        sujet = _date_sujet(it["prompt"])
        sents = _sentences_dates_v3(sujet, it["reponses_acceptees"][0])
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]
        attach({"id": it["id"], "categorie": "dates"},
               sents)
    for i, (sujet, annee) in enumerate(F2.DATES_EXTRA, start=1):
        attach({"id": f"dat-x{i:03d}", "categorie": "dates"},
               _sentences_dates_v3(sujet, annee))

    for it in _probe_pc_items():
        if it["categorie"] != "definitions":
            continue
        terme_art, nu, _kw, complement = F2.DEFINITIONS_PROBE[it["id"]]
        sents = _sentences_definitions_v3(terme_art, nu, complement)
        assert any(s.startswith(it["prompt"]) for s in sents), it["id"]
        attach({"id": it["id"], "categorie": "definitions"}, sents)
    for i, (terme_art, nu, _kw, complement) in enumerate(
            F2.DEFINITIONS_EXTRA, start=1):
        attach({"id": f"def-x{i:03d}", "categorie": "definitions"},
               _sentences_definitions_v3(terme_art, nu, complement))

    for f in unis:
        items = list(UNIT_SENTENCES[f["id"]])
        sujet, verbe, val = UNIT_TRIPLES[f["id"]]
        items += _sentences_units_v3(sujet, verbe, val, "")
        assert any(s.startswith(f["prompt"]) for s in items), f["id"]
        attach({"id": f["id"], "categorie": "unites_sciences"}, items)
    for i, (sujet, verbe, a, unite, _kw) in enumerate(F2.UNITS_EXTRA, start=1):
        attach({"id": f"uni-x{i:03d}", "categorie": "unites_sciences"},
               _sentences_units_v3(sujet, verbe, a, unite))

    for f in facts:
        assert len(f["paragraphs"]) == N_VARIANTS_V3, f["id"]
        assert len(set(f["paragraphs"])) == N_VARIANTS_V3, f["id"]
    assert len(facts) == 534, len(facts)
    return facts


# ---------------------------------------------------------------------------
# Garde-fous : aucune formulation `reformule` ne doit fuiter.
# ---------------------------------------------------------------------------

def _norm(t: str) -> str:
    d = unicodedata.normalize("NFD", t)
    d = "".join(c for c in d if not unicodedata.combining(c))
    d = "".join(c if (c.isalnum() or c.isspace()) else " " for c in d.lower())
    return " ".join(d.split())


def reformule_guards(categories=GUARD_CATS_V1) -> list[str]:
    items = [json.loads(l) for l in open(PROBE, encoding="utf-8") if l.strip()]
    guards = []
    for it in items:
        if it["slice"] != "reformule":
            continue
        if it["categorie"] not in categories:
            continue
        g = _norm(it["prompt"])
        # On retire le point d'interrogation déjà transformé en espace ;
        # la sous-chaîne testée est la formulation complète.
        if g:
            guards.append(g)
    return guards


def assert_no_leak(texts: list[str], allow_questions: bool = False,
                   categories=GUARD_CATS_V1) -> None:
    """Garde anti-fuite : aucune formulation `reformule` de la sonde.

    `allow_questions=False` pour le QA généré (templates déclaratifs only) ;
    le texte FR général, lui, contient légitimement des « ? ».
    `categories` sélectionne les catégories gardées (v1 : capitales+unites ;
    v2 : les 4).
    """
    guards = reformule_guards(categories)
    for t in texts:
        if not allow_questions:
            assert "?" not in t, f"interrogation interdite : {t[:80]!r}"
        nt = _norm(t)
        for g in guards:
            if g and g in nt:
                raise AssertionError(
                    f"FUITE d'une formulation reformule : {g!r} dans {t[:120]!r}")


# ---------------------------------------------------------------------------
# Texte général FR (déterministe).
# ---------------------------------------------------------------------------

def load_general_docs(limit: int) -> list[str]:
    sys.path.insert(0, ROOT)
    import data as D

    texts = D.load_texts(GENERAL)
    idx = list(range(len(texts)))
    random.Random(GENERAL_SEED).shuffle(idx)
    out = [texts[i] for i in idx[:limit]]
    return out


# ---------------------------------------------------------------------------
# Assemblage + mesure en tokens (tokenizer parent, CPU).
# ---------------------------------------------------------------------------

def _stream_len(tokenizer, texts: list[str]) -> int:
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) + 1
               for t in texts)


def _windows(stream_len: int) -> int:
    return max(0, (stream_len - 1) // 128)


def build_docs(repeat: int = REPEAT, target_steps: int = SPE_TARGET,
               general_ratio: float = 0.07, facts: list[dict] | None = None,
               guard_categories=GUARD_CATS_V1, general_limit: int = 400,
               general_doc_max_tok: int = GENERAL_DOC_MAX_TOK) -> dict:
    """Assemble QA (r-major, donc non adjacent) + FR général (ratio demandé).

    Cible exprimée en **steps optimiseur/époque** : 1 step = save_batch 8
    x accum 16 x seq 128 = 16 384 tokens. SPE_TARGET=379 (v1) / 413 (v2)
    donne une époque neuve depuis le step 22740 de t08.

    `facts=None` -> `build_facts()` (v1, comportement gelé). Les défauts
    reproduisent la v1 à l'identique ; la v2 fournit `facts`, `guard_categories`
    (4 catégories) et des limites général élargies.

    Retourne un dict {docs, n_qa, n_general, stream, windows, tokens}.
    """
    from main import load_tokenizer

    tok = load_tokenizer(PARENT)
    facts = build_facts() if facts is None else facts
    qa: list[str] = []
    for _r in range(repeat):
        for f in facts:
            qa.extend(f["paragraphs"])
    assert_no_leak(qa, categories=guard_categories)

    cache: dict[str, int] = {}

    def n_tok(t: str) -> int:
        if t not in cache:
            cache[t] = len(tok.encode(t, add_special_tokens=False)) + 1
        return cache[t]

    # +1 : make_batches donne floor((L-1)/128) fenêtres ; on veut
    # exactement target_steps * 128 fenêtres.
    target = target_steps * 16384 + 1
    qa_target = int(target * (1 - general_ratio))

    docs: list[str] = []
    stream = 0
    for t in qa:
        n = n_tok(t)
        if stream + n > qa_target:
            break
        docs.append(t)
        stream += n
    n_qa = len(docs)
    qa_stream = stream

    # Texte général : docs plafonnés, anti-fuite, tronqués au budget restant.
    general = load_general_docs(general_limit)
    assert_no_leak(general, allow_questions=True, categories=guard_categories)
    n_general = 0
    for g in general:
        room = target - stream
        if room <= 200:
            break
        ids = tok.encode(g, add_special_tokens=False)[:general_doc_max_tok]
        if len(ids) + 1 > room:
            ids = ids[: room - 1]
        docs.append(tok.decode(ids))
        stream += len(ids) + 1
        n_general += 1

    # Ajustement fin au paragraphe QA (granularité ~1 fenêtre).
    i = 0
    while stream < target - 200:
        t = qa[i % len(qa)]
        docs.append(t)
        stream += n_tok(t)
        n_qa += 1
        i += 1

    general_stream = stream - qa_stream
    windows = max(0, (stream - 1) // 128)
    assert windows == target_steps * 128, (windows, target_steps * 128)
    return {"docs": docs, "n_qa": n_qa, "n_general": n_general,
            "stream": stream, "windows": windows, "tokens": windows * 128,
            "qa_tokens": qa_stream, "general_tokens": general_stream}


def build_docs_v3(facts: list[dict], repeat: int = REPEAT_V3,
                  target_steps: int = SPE_TARGET_V3,
                  general_ratio: float = GENERAL_RATIO_V3,
                  general_limit: int = GENERAL_LIMIT_V3,
                  general_doc_max_tok: int = GENERAL_DOC_MAX_TOK_V3) -> dict:
    """Assemble v3 : TOUTES les répétitions QA (aucune troncature) + général.

    Contrairement aux v1/v2, le QA est conservé intégralement (534 faits x 40
    variantes x 10 répétitions) : on vérifie qu'il tient sous `1 - ratio`,
    puis le FR général remplit EXACTEMENT la cible (dernier doc tronqué).
    """
    from main import load_tokenizer

    tok = load_tokenizer(PARENT)
    qa: list[str] = []
    for _r in range(repeat):
        for f in facts:
            qa.extend(f["paragraphs"])
    assert_no_leak(qa, categories=GUARD_CATS_ALL)

    cache: dict[str, int] = {}

    def n_tok(t: str) -> int:
        if t not in cache:
            cache[t] = len(tok.encode(t, add_special_tokens=False)) + 1
        return cache[t]

    target = target_steps * 16384 + 1
    qa_target = int(target * (1 - general_ratio))
    qa_stream = sum(n_tok(t) for t in qa)
    assert qa_stream <= qa_target, (qa_stream, qa_target)

    docs = list(qa)
    stream = qa_stream
    general = load_general_docs(general_limit)
    assert_no_leak(general, allow_questions=True, categories=GUARD_CATS_ALL)
    n_general = 0
    gi = 0
    while stream < target:
        room = target - stream
        if room <= 1:
            break
        g = general[gi % len(general)]
        gi += 1
        ids = tok.encode(g, add_special_tokens=False)[:general_doc_max_tok]
        if len(ids) + 1 > room:
            ids = ids[: room - 1]
        if not ids:
            break
        docs.append(tok.decode(ids))
        stream += len(ids) + 1
        n_general += 1
    assert stream == target, (stream, target)
    windows = (stream - 1) // 128
    assert windows == target_steps * 128, (windows, target_steps * 128)
    return {"docs": docs, "n_qa": len(qa), "n_general": n_general,
            "stream": stream, "windows": windows, "tokens": windows * 128,
            "qa_tokens": qa_stream, "general_tokens": stream - qa_stream,
            "n_real": sum(len(f["real_sentences"]) for f in facts),
            "n_facts_real": sum(1 for f in facts if f["real_sentences"])}



def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Corpus QA gelé du pilote.")
    ap.add_argument("--version", default="v1", choices=["v1", "v2", "v3"])
    ap.add_argument("--repeat", type=int, default=None)
    ap.add_argument("--target-steps", type=int, default=None)
    ap.add_argument("--general-ratio", type=float, default=None)
    ap.add_argument("--out-jsonl", default="")
    ap.add_argument("--out-manifest", default="")
    a = ap.parse_args(argv)

    if a.version == "v1":
        repeat = a.repeat if a.repeat is not None else REPEAT
        target_steps = (a.target_steps if a.target_steps is not None
                        else SPE_TARGET)
        general_ratio = (a.general_ratio if a.general_ratio is not None
                         else 0.07)
        facts = build_facts()
        n_variants = N_VARIANTS
        cats = ("capitales", "unites_sciences")
        templates = {
            "capitales": FRAMES_CAPITALES,
            "unites_sciences": "phrases déclaratives par fait (UNIT_SENTENCES)",
        }
        general_limit, doc_max = 400, GENERAL_DOC_MAX_TOK
        guard_cats = GUARD_CATS_V1
        out_jsonl = a.out_jsonl or OUT_JSONL
        out_manifest = a.out_manifest or OUT_MANIFEST
        version, gel = VERSION, ("Corpus figé AVANT entraînement : hash SHA256 "
                                 "ci-dessus. Toute évolution = qa_fr_pilot_v2.")
    elif a.version == "v2":
        repeat = a.repeat if a.repeat is not None else REPEAT_V2
        target_steps = (a.target_steps if a.target_steps is not None
                        else SPE_TARGET_V2)
        general_ratio = (a.general_ratio if a.general_ratio is not None
                         else GENERAL_RATIO_V2)
        facts = build_facts_v2()
        n_variants = N_VARIANTS_V2
        cats = CATEGORIES_V2
        templates = {
            "capitales": FRAMES_CAPITALES,
            "dates": FRAMES_DATES,
            "definitions": FRAMES_DEFINITIONS,
            "unites_sciences": ("phrases déclaratives des 20 unités de la sonde "
                                "(UNIT_SENTENCES) + trames génériques"),
        }
        general_limit, doc_max = GENERAL_LIMIT_V2, GENERAL_DOC_MAX_TOK_V2
        guard_cats = GUARD_CATS_ALL
        out_jsonl = a.out_jsonl or OUT_JSONL_V2
        out_manifest = a.out_manifest or OUT_MANIFEST_V2
        version, gel = VERSION_V2, ("Corpus figé AVANT entraînement : hash "
                                    "SHA256 ci-dessus. Toute évolution = "
                                    "qa_fr_pilot_v3.")
    else:
        repeat = a.repeat if a.repeat is not None else REPEAT_V3
        target_steps = (a.target_steps if a.target_steps is not None
                        else SPE_TARGET_V3)
        general_ratio = (a.general_ratio if a.general_ratio is not None
                         else GENERAL_RATIO_V3)
        facts = build_facts_v3()
        n_variants = N_VARIANTS_V3
        cats = CATEGORIES_V2
        templates = {
            "capitales": FRAMES_CAPITALES_V3,
            "dates": FRAMES_DATES_V3,
            "definitions": FRAMES_DEFINITIONS_V3,
            "unites_sciences": (list(FRAMES_UNITS_V3)
                                + ["UNIT_SENTENCES (20 unités de la sonde)"]),
            "phrases_reelles": ("tranche_05 : phrases contenant l'entité et la "
                                "réponse (sous-chaîne, frontières de tokens), "
                                f"max {REAL_MAX_V3}/fait"),
        }
        general_limit, doc_max = GENERAL_LIMIT_V3, GENERAL_DOC_MAX_TOK_V3
        guard_cats = GUARD_CATS_ALL
        out_jsonl = a.out_jsonl or OUT_JSONL_V3
        out_manifest = a.out_manifest or OUT_MANIFEST_V3
        version, gel = VERSION_V3, ("Corpus figé AVANT entraînement : hash "
                                    "SHA256 ci-dessus. Toute évolution = "
                                    "qa_fr_pilot_v4.")
    if a.version == "v3":
        res = build_docs_v3(facts, repeat=repeat, target_steps=target_steps,
                            general_ratio=general_ratio,
                            general_limit=general_limit,
                            general_doc_max_tok=doc_max)
    else:
        res = build_docs(repeat=repeat, target_steps=target_steps,
                         general_ratio=general_ratio, facts=facts,
                         guard_categories=guard_cats,
                         general_limit=general_limit,
                         general_doc_max_tok=doc_max)
    docs = res["docs"]
    n_qa, n_general = res["n_qa"], res["n_general"]
    stream, wins, tokens = res["stream"], res["windows"], res["tokens"]
    part_general = res["general_tokens"] / max(stream, 1)
    spe = (wins + 127) // 128  # 1 step = 8 x 16 = 128 fenêtres

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in docs:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    blob = open(out_jsonl, "rb").read()
    sha = hashlib.sha256(blob).hexdigest()

    manifest = {
        "version": version,
        "fichier": os.path.relpath(out_jsonl, ROOT),
        "sha256": sha,
        "octets": len(blob),
        "n_docs": len(docs),
        "n_docs_qa": n_qa,
        "n_docs_general": n_general,
        "part_general_docs": round(n_general / len(docs), 4),
        "part_general_tokens": round(part_general, 4),
        "tokens_qa": res["qa_tokens"],
        "tokens_general": res["general_tokens"],
        "stream_tokens": stream,
        "tokens_fenetres": tokens,
        "steps_par_epoque": spe,
        "tokens_par_step": 16384,
        "repeat": repeat,
        "n_variants_par_fait": n_variants,
        "n_faits": len(facts),
        "faits": {c: sum(1 for f in facts if f["categorie"] == c)
                  for c in cats},
        "templates": templates,
        "interdit": ("aucune formulation de la slice `reformule` de la sonde ; "
                     "aucun « ? » dans le QA généré ; vérifié à la génération"),
        "general": {"source": GENERAL, "seed": GENERAL_SEED,
                    "doc_max_tokens": doc_max,
                    "mode": "permutation déterministe, préfixe, docs plafonnés"},
        "split": {"train": "tous les docs ci-dessus",
                  "val": "runs/abl_data/val_E_frozen.jsonl (gelée, externe)"},
        "probe_sha256": hashlib.sha256(open(PROBE, "rb").read()).hexdigest(),
        "generateur": os.path.relpath(os.path.abspath(__file__), ROOT),
        "generateur_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
        "gel": gel,
    }
    if a.version == "v3":
        manifest["corpus_reel"] = {
            "source": GENERAL,
            "mode": ("phrases contenant l'entité ET la réponse ; sous-chaîne "
                     "avec frontières de tokens ; ordre du fichier"),
            "max_par_fait": REAL_MAX_V3,
            "longueur_min": REAL_MIN_CHARS_V3,
            "longueur_max": REAL_MAX_CHARS_V3,
            "n_phrases": res["n_real"],
            "n_faits_avec_phrase": res["n_facts_real"],
            "filtre_capitales": "mot « capitale » exigé",
        }
        manifest["epochs_cumulatives"] = {
            "prev_steps": 22740,
            "spe": spe,
            "e_cumul": 22740 // spe + 1,
            "steps_restants": (22740 // spe + 1) * spe - 22740,
        }
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")

    print(f"[qa] docs={len(docs)} (qa {n_qa} + general {n_general} = "
          f"{100 * n_general / len(docs):.1f} % des docs)")
    print(f"[qa] stream={stream} tokens, fenetres={wins}, "
          f"tokens_fenetres={tokens}, SPE={spe}, general={part_general:.2%}")
    print(f"[qa] repeat={repeat} x {n_variants} variantes x {len(facts)} faits")
    print(f"[qa] sha256 {sha}")
    print(f"[qa] -> {os.path.relpath(out_jsonl, ROOT)} + manifeste")


if __name__ == "__main__":
    main()
