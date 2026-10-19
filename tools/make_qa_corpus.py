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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Corpus QA gelé du pilote.")
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
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
    else:
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

    res = build_docs(repeat=repeat, target_steps=target_steps,
                     general_ratio=general_ratio, facts=facts,
                     guard_categories=guard_cats, general_limit=general_limit,
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
