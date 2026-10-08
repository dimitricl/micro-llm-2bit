"""Regenere data_clean/monde_fr.txt avec noms FR (pays + capitales).

Sources : /Volumes/Lexar/rag-data/monde/*.txt (4 lignes : Pays, Capitale,
Région, Population). Les noms anglais sont traduits via les tables
PAYS_FR / CAPS_FR ci-dessous ; toute entrée absente des tables est
écrite dans data_clean/monde_missing.txt et IGNORÉE (jamais inventée).

Template (continuité avec l'ancien fichier) :
    "{P} est un pays {prep} dont la capitale est {C}."
Imperfection connue : genre uniforme ("un pays") même pour pays féminins.
"""

import json
import os
import re

SRC = "/Volumes/Lexar/rag-data/monde"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(OUT_DIR, "monde_fr.txt")
MISSING = os.path.join(OUT_DIR, "monde_missing.txt")

PAYS_FR = {
    "Afghanistan": "Afghanistan",
    "Albania": "Albanie",
    "Algeria": "Algérie",
    "American Samoa": "Samoa américaines",
    "Andorra": "Andorre",
    "Angola": "Angola",
    "Anguilla": "Anguilla",
    "Antarctica": "Antarctique",
    "Antigua and Barbuda": "Antigua-et-Barbuda",
    "Argentina": "Argentine",
    "Armenia": "Arménie",
    "Aruba": "Aruba",
    "Australia": "Australie",
    "Austria": "Autriche",
    "Azerbaijan": "Azerbaïdjan",
    "Bahamas": "Bahamas",
    "Bahrain": "Bahreïn",
    "Bangladesh": "Bangladesh",
    "Barbados": "Barbade",
    "Belarus": "Biélorussie",
    "Belgium": "Belgique",
    "Belize": "Belize",
    "Benin": "Bénin",
    "Bermuda": "Bermudes",
    "Bhutan": "Bhoutan",
    "Bolivia": "Bolivie",
    "Bosnia and Herzegovina": "Bosnie-Herzégovine",
    "Botswana": "Botswana",
    "Bouvet Island": "Île Bouvet",
    "Brazil": "Brésil",
    "British Indian Ocean Territory": "Territoire britannique de l'océan Indien",
    "British Virgin Islands": "Îles Vierges britanniques",
    "Brunei": "Brunei",
    "Bulgaria": "Bulgarie",
    "Burkina Faso": "Burkina Faso",
    "Burundi": "Burundi",
    "Cambodia": "Cambodge",
    "Cameroon": "Cameroun",
    "Canada": "Canada",
    "Cape Verde": "Cap-Vert",
    "Caribbean Netherlands": "Pays-Bas caribéens",
    "Cayman Islands": "Îles Caïmans",
    "Central African Republic": "République centrafricaine",
    "Chad": "Tchad",
    "Chile": "Chili",
    "China": "Chine",
    "Christmas Island": "Île Christmas",
    "Cocos (Keeling) Islands": "Îles Cocos",
    "Colombia": "Colombie",
    "Comoros": "Comores",
    "Congo": "Congo",
    "Cook Islands": "Îles Cook",
    "Costa Rica": "Costa Rica",
    "Croatia": "Croatie",
    "Cuba": "Cuba",
    "Curaçao": "Curaçao",
    "Cyprus": "Chypre",
    "Czechia": "Tchéquie",
    "DR Congo": "République démocratique du Congo",
    "Denmark": "Danemark",
    "Djibouti": "Djibouti",
    "Dominica": "Dominique",
    "Dominican Republic": "République dominicaine",
    "Ecuador": "Équateur",
    "Egypt": "Égypte",
    "El Salvador": "Salvador",
    "Equatorial Guinea": "Guinée équatoriale",
    "Eritrea": "Érythrée",
    "Estonia": "Estonie",
    "Eswatini": "Eswatini",
    "Ethiopia": "Éthiopie",
    "Falkland Islands": "Îles Malouines",
    "Faroe Islands": "Îles Féroé",
    "Fiji": "Fidji",
    "Finland": "Finlande",
    "France": "France",
    "French Guiana": "Guyane",
    "French Polynesia": "Polynésie française",
    "French Southern and Antarctic Lands": "Terres australes et antarctiques françaises",
    "Gabon": "Gabon",
    "Gambia": "Gambie",
    "Georgia": "Géorgie",
    "Germany": "Allemagne",
    "Ghana": "Ghana",
    "Gibraltar": "Gibraltar",
    "Greece": "Grèce",
    "Greenland": "Groenland",
    "Grenada": "Grenade",
    "Guadeloupe": "Guadeloupe",
    "Guam": "Guam",
    "Guatemala": "Guatemala",
    "Guernsey": "Guernesey",
    "Guinea": "Guinée",
    "Guinea-Bissau": "Guinée-Bissau",
    "Guyana": "Guyana",
    "Haiti": "Haïti",
    "Heard Island and McDonald Islands": "Îles Heard-et-MacDonald",
    "Honduras": "Honduras",
    "Hong Kong": "Hong Kong",
    "Hungary": "Hongrie",
    "Iceland": "Islande",
    "India": "Inde",
    "Indonesia": "Indonésie",
    "Iran": "Iran",
    "Iraq": "Irak",
    "Ireland": "Irlande",
    "Isle of Man": "Île de Man",
    "Israel": "Israël",
    "Italy": "Italie",
    "Ivory Coast": "Côte d'Ivoire",
    "Jamaica": "Jamaïque",
    "Japan": "Japon",
    "Jersey": "Jersey",
    "Jordan": "Jordanie",
    "Kazakhstan": "Kazakhstan",
    "Kenya": "Kenya",
    "Kiribati": "Kiribati",
    "Kosovo": "Kosovo",
    "Kuwait": "Koweït",
    "Kyrgyzstan": "Kirghizistan",
    "Laos": "Laos",
    "Latvia": "Lettonie",
    "Lebanon": "Liban",
    "Lesotho": "Lesotho",
    "Liberia": "Liberia",
    "Libya": "Libye",
    "Liechtenstein": "Liechtenstein",
    "Lithuania": "Lituanie",
    "Luxembourg": "Luxembourg",
    "Macau": "Macao",
    "Madagascar": "Madagascar",
    "Malawi": "Malawi",
    "Malaysia": "Malaisie",
    "Maldives": "Maldives",
    "Mali": "Mali",
    "Malta": "Malte",
    "Marshall Islands": "Îles Marshall",
    "Martinique": "Martinique",
    "Mauritania": "Mauritanie",
    "Mauritius": "Maurice",
    "Mayotte": "Mayotte",
    "Mexico": "Mexique",
    "Micronesia": "Micronésie",
    "Moldova": "Moldavie",
    "Monaco": "Monaco",
    "Mongolia": "Mongolie",
    "Montenegro": "Monténégro",
    "Montserrat": "Montserrat",
    "Morocco": "Maroc",
    "Mozambique": "Mozambique",
    "Myanmar": "Birmanie",
    "Namibia": "Namibie",
    "Nauru": "Nauru",
    "Nepal": "Népal",
    "Netherlands": "Pays-Bas",
    "New Caledonia": "Nouvelle-Calédonie",
    "New Zealand": "Nouvelle-Zélande",
    "Nicaragua": "Nicaragua",
    "Niger": "Niger",
    "Nigeria": "Nigeria",
    "Niue": "Niue",
    "Norfolk Island": "Île Norfolk",
    "North Korea": "Corée du Nord",
    "North Macedonia": "Macédoine du Nord",
    "Northern Mariana Islands": "Îles Mariannes du Nord",
    "Norway": "Norvège",
    "Oman": "Oman",
    "Pakistan": "Pakistan",
    "Palau": "Palaos",
    "Palestine": "Palestine",
    "Panama": "Panama",
    "Papua New Guinea": "Papouasie-Nouvelle-Guinée",
    "Paraguay": "Paraguay",
    "Peru": "Pérou",
    "Philippines": "Philippines",
    "Pitcairn Islands": "Îles Pitcairn",
    "Poland": "Pologne",
    "Portugal": "Portugal",
    "Puerto Rico": "Porto Rico",
    "Qatar": "Qatar",
    "Romania": "Roumanie",
    "Russia": "Russie",
    "Rwanda": "Rwanda",
    "Réunion": "La Réunion",
    "Saint Barthélemy": "Saint-Barthélemy",
    "Saint Helena, Ascension and Tristan da Cunha": "Sainte-Hélène, Ascension et Tristan da Cunha",
    "Saint Kitts and Nevis": "Saint-Christophe-et-Niévès",
    "Saint Lucia": "Sainte-Lucie",
    "Saint Martin": "Saint-Martin",
    "Saint Pierre and Miquelon": "Saint-Pierre-et-Miquelon",
    "Saint Vincent and the Grenadines": "Saint-Vincent-et-les-Grenadines",
    "Samoa": "Samoa",
    "San Marino": "Saint-Marin",
    "Sao Tome and Principe": "Sao Tomé-et-Principe",
    "São Tomé and Príncipe": "Sao Tomé-et-Principe",
    "Saudi Arabia": "Arabie saoudite",
    "Senegal": "Sénégal",
    "Serbia": "Serbie",
    "Seychelles": "Seychelles",
    "Sierra Leone": "Sierra Leone",
    "Singapore": "Singapour",
    "Sint Maarten": "Saint-Martin (partie néerlandaise)",
    "Slovakia": "Slovaquie",
    "Slovenia": "Slovénie",
    "Solomon Islands": "Îles Salomon",
    "Somalia": "Somalie",
    "South Africa": "Afrique du Sud",
    "South Georgia": "Géorgie du Sud",
    "South Georgia and the South Sandwich Islands": "Géorgie du Sud-et-les îles Sandwich du Sud",
    "South Korea": "Corée du Sud",
    "South Sudan": "Soudan du Sud",
    "Spain": "Espagne",
    "Sri Lanka": "Sri Lanka",
    "Sudan": "Soudan",
    "Suriname": "Suriname",
    "Svalbard and Jan Mayen": "Svalbard et Jan Mayen",
    "Sweden": "Suède",
    "Switzerland": "Suisse",
    "Syria": "Syrie",
    "Taiwan": "Taïwan",
    "Tajikistan": "Tadjikistan",
    "Tanzania": "Tanzanie",
    "Thailand": "Thaïlande",
    "Timor-Leste": "Timor oriental",
    "Togo": "Togo",
    "Tokelau": "Tokelau",
    "Tonga": "Tonga",
    "Trinidad and Tobago": "Trinité-et-Tobago",
    "Tunisia": "Tunisie",
    "Turkey": "Turquie",
    "Türkiye": "Turquie",
    "Turkmenistan": "Turkménistan",
    "Turks and Caicos Islands": "Îles Turques-et-Caïques",
    "Tuvalu": "Tuvalu",
    "Uganda": "Ouganda",
    "Ukraine": "Ukraine",
    "United Arab Emirates": "Émirats arabes unis",
    "United Kingdom": "Royaume-Uni",
    "United States": "États-Unis",
    "United States Virgin Islands": "Îles Vierges des États-Unis",
    "United States Minor Outlying Islands": "Îles mineures éloignées des États-Unis",
    "Uruguay": "Uruguay",
    "Uzbekistan": "Ouzbékistan",
    "Vanuatu": "Vanuatu",
    "Vatican City": "Vatican",
    "Venezuela": "Venezuela",
    "Vietnam": "Viêt Nam",
    "Virgin Islands (US)": "Îles Vierges des États-Unis",
    "Wallis and Futuna": "Wallis-et-Futuna",
    "Western Sahara": "Sahara occidental",
    "Yemen": "Yémen",
    "Zambia": "Zambie",
    "Zimbabwe": "Zimbabwe",
    "Åland": "Åland",
    "Åland Islands": "Åland",
}

# Capitales qui diffèrent entre anglais et français (les autres sont identiques).
CAPS_FR = {
    "Abu Dhabi": "Abou Dabi",
    "Algiers": "Alger",
    "Ashgabat": "Achgabat",
    "Beijing": "Pékin",
    "Beirut": "Beyrouth",
    "Bishkek": "Bichkek",
    "Bogotá": "Bogota",
    "Brasília": "Brasilia",
    "Bucharest": "Bucarest",
    "Cairo": "Le Caire",
    "Copenhagen": "Copenhague",
    "Damascus": "Damas",
    "Dhaka": "Dacca",
    "Dushanbe": "Douchanbé",
    "El Aaiún": "Laâyoune",
    "Guatemala City": "Guatemala",
    "Havana": "La Havane",
    "Jerusalem": "Jérusalem",
    "Kabul": "Kaboul",
    "Kyiv": "Kiev",
    "Kuwait City": "Koweït",
    "London": "Londres",
    "Mexico City": "Mexico",
    "Moscow": "Moscou",
    "Nicosia": "Nicosie",
    "Panama City": "Panama",
    "Port of Spain": "Port-d'Espagne",
    "Riyadh": "Riyad",
    "Sana'a": "Sanaa",
    "Santo Domingo": "Saint-Domingue",
    "Seoul": "Séoul",
    "Singapore": "Singapour",
    "St. George's": "Saint-Georges",
    "Tashkent": "Tachkent",
    "Tbilisi": "Tbilissi",
    "Tehran": "Téhéran",
    "Ulan Bator": "Oulan-Bator",
    "Valletta": "La Valette",
    "Vatican City": "Vatican",
    "Vienna": "Vienne",
    "Vientiane": "Vientiane",
    "Warsaw": "Varsovie",
    "Washington D.C.": "Washington",
    "Yerevan": "Erevan",
}

REGION_PREP = {
    "Africa": "d'Afrique",
    "Americas": "des Amériques",
    "Antarctic": "en Antarctique",
    "Asia": "d'Asie",
    "Europe": "d'Europe",
    "Oceania": "d'Océanie",
}

CAP_RE = re.compile(r"""Capitale\s*:\s*\[\s*(?:"([^"]+)"|'([^']+)'|([^\]]+?))\s*\]""")


def _cap_value(m: re.Match) -> str:
    return next(g for g in m.groups() if g is not None).strip()
PAYS_RE = re.compile(r"Pays\s*:\s*(.+)")
REG_RE = re.compile(r"Région\s*:\s*(.+)")


def main() -> None:
    phrases, missing = [], []
    for fname in sorted(os.listdir(SRC)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(SRC, fname), encoding="utf-8", errors="replace") as f:
            d = f.read()
        m_p, m_c, m_r = PAYS_RE.search(d), CAP_RE.search(d), REG_RE.search(d)
        if not (m_p and m_c):
            missing.append({"fichier": fname, "raison": "pays ou capitale illisible"})
            continue
        p_en, c_en = m_p.group(1).strip(), _cap_value(m_c)
        r_en = m_r.group(1).strip() if m_r else ""
        p_fr = PAYS_FR.get(p_en)
        c_fr = CAPS_FR.get(c_en, c_en)  # capitales souvent identiques
        if p_fr is None:
            missing.append({"fichier": fname, "raison": f"pays sans traduction : {p_en}"})
            continue
        if c_en not in CAPS_FR:
            # Vérifie que la capitale "identique" est plausible (pas de résidu).
            if not c_en or len(c_en) < 2:
                missing.append({"fichier": fname, "raison": f"capitale vide : {c_en}"})
                continue
        prep = REGION_PREP.get(r_en, "")
        phrase = f"{p_fr} est un pays {prep} dont la capitale est {c_fr}."
        phrases.append(re.sub(r"\s+", " ", phrase).strip())
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(phrases) + "\n")
    with open(MISSING, "w", encoding="utf-8") as f:
        for m in missing:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[monde] {len(phrases)} phrases -> {OUT}, {len(missing)} manquantes -> {MISSING}")
    for m in missing:
        print("  MANQUANT :", m)


if __name__ == "__main__":
    main()
