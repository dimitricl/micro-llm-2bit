"""Genere des traces <think> via Ollama (modele raisonnant) -> JSONL.

Usage : python gen_think.py <in_questions.jsonl> <out.jsonl> [max_n]
Chaque ligne in : {"q": "...", "contexte": "...", "attendu": "..."?}.
Sortie : {"text": "Question : ...\\n<think>...</think>\\nReponse : ..."}.
Rejets : <out>.rejected.jsonl avec {"q": ..., "raison": ...}.

Parsing strict (regex) + validation + 1 relance ciblée :
- réponse vide (len=0, serveur fatigué) -> 1 relance après pause ;
- ResponseError "tool call" (le client parse le brut comme un appel
  d'outil, vu avec gpt-oss) -> 1 relance en texte brut, température basse ;
- think tronqué (pas de </think> ou pas de 'Reponse :') -> 1 relance en
  demandant un raisonnement court.
Options conservées : num_predict=768, reprise en append, logs
ratee/ignoree, compteur skipped, THINK_MODEL (défaut phi4-mini).
"""

import json
import os
import re
import sys
import time

def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    # Reprise : n'écrase plus le fichier existant, on complète jusqu'à max_n.
    done = 0
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f_done:
            for line in f_done:
                if line.strip():
                    done += 1
    print(f"[think] reprise: {done} traces déjà dans {out}", flush=True)

    rej_path = out + ".rejected.jsonl"
    n = done
    skipped = 0
    with open(src, encoding="utf-8") as f_in, \
            open(out, "a", encoding="utf-8") as f_out, \
            open(rej_path, "a", encoding="utf-8") as f_rej:
        for i, line in enumerate(f_in):
            if i < done:  # déjà générées lors d'un run précédent
                continue
            if n >= max_n:
                break
            row = json.loads(line)
            texte, raison = genere_avec_relance(
                row["q"], row.get("contexte", ""), row.get("attendu", "")
            )
            if raison:
                print(f"[think] ignoree {n}: {raison}", flush=True)
                f_rej.write(json.dumps({"q": row["q"], "raison": raison},
                                       ensure_ascii=False) + "\n")
                skipped += 1
                continue
            f_out.write(json.dumps({"text": texte}, ensure_ascii=False) + "\n")
            n += 1
            f_out.flush()
            if n % 20 == 0:
                print(f"[think] {n} traces (skipped={skipped})", flush=True)
    print(f"[think] TERMINE : {n} traces -> {out} (skipped={skipped})")

import ollama  # lit OLLAMA_HOST
from ollama import ResponseError

MODEL = os.environ.get("THINK_MODEL", "phi4-mini")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
REPONSE_RE = re.compile(r"^Reponse\s*:\s*(.+)", re.MULTILINE | re.DOTALL)

# Heuristique "français" : mots-outils courants (au moins 3 distincts).
MOTS_FR = {
    "le", "la", "les", "de", "des", "du", "un", "une", "est", "sont",
    "et", "dans", "pour", "avec", "qui", "que", "pas", "plus", "cette",
    "ces", "son", "sa", "par", "sur", "comme",
}


def parse_strict(txt: str) -> tuple[str, str] | None:
    """Extrait (think, reponse) ou None si format non conforme."""
    mt = THINK_RE.search(txt)
    mr = REPONSE_RE.search(txt)
    if not mt or not mr:
        return None
    think, rep = mt.group(1).strip(), mr.group(1).strip().split("\n")[0].strip()
    if not think or not rep:
        return None
    return think, rep


def est_francais(think: str) -> bool:
    mots = set(re.findall(r"[a-zàâäéèêëîïôöùûüç]+", think.lower()))
    return len(mots & MOTS_FR) >= 3


def valide(think: str, rep: str, attendu: str = "") -> str:
    """Retourne '' si OK, sinon la raison du rejet."""
    if not (20 <= len(think) <= 2000):
        return f"think_longueur_{len(think)}"
    if not (2 <= len(rep) <= 500):
        return f"reponse_longueur_{len(rep)}"
    if not est_francais(think):
        return "think_pas_francais"
    if attendu and attendu.lower() not in rep.lower():
        return "reponse_ne_contient_pas_attendu"
    return ""


def appelle(prompt: str, temperature: float = 0.6) -> str:
    res = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": temperature, "num_predict": 768},
    )
    return (res.get("response") or "").strip()


BASE_CONSIGNE = (
    "Reponds en francais. D'abord ton raisonnement etape par etape "
    "entre les balises <think> et </think>, puis la reponse finale "
    "sur une ligne commencant par 'Reponse : '."
)


def genere_avec_relance(q: str, ctx: str, attendu: str = "") -> tuple[str, str]:
    """Retourne (texte_final, '') ou ('', raison_rejet). Une seule relance."""
    prompt = f"Contexte : {ctx}\nQuestion : {q}\n{BASE_CONSIGNE}"
    try:
        txt = appelle(prompt)
    except ResponseError as e:
        if "tool call" in str(e).lower():
            # Le serveur a parsé le brut comme un appel d'outil : relance
            # en texte brut, température basse.
            time.sleep(5)
            try:
                txt = appelle(
                    prompt + " Reponds en texte brut, sans aucun appel d'outil.",
                    temperature=0.3,
                )
            except Exception as e2:
                return "", f"ratee_toolcall_apres_relance:{type(e2).__name__}"
        else:
            return "", f"ratee_response_error:{e}"
    except Exception as e:
        return "", f"ratee_{type(e).__name__}"
    if not txt:
        # Réponse vide (serveur fatigué) : pause puis 1 relance.
        time.sleep(10)
        try:
            txt = appelle(prompt)
        except Exception as e:
            return "", f"ratee_vide_apres_relance:{type(e).__name__}"
        if not txt:
            return "", "reponse_vide"
    parsed = parse_strict(txt)
    if parsed is None:
        # Think tronqué ou consigne ignorée : 1 relance en court.
        time.sleep(2)
        try:
            txt2 = appelle(
                f"Contexte : {ctx}\nQuestion : {q}\n"
                "Reponds en francais avec un raisonnement COURT (5 etapes max) "
                "entre <think> et </think>, puis 'Reponse : ' et la reponse."
            )
        except Exception as e:
            return "", f"ratee_relance_courte:{type(e).__name__}"
        parsed = parse_strict(txt2 or "")
        if parsed is None:
            return "", f"pas_de_balises_apres_relance_len_{len(txt2 or '')}"
        txt = txt2
    think, rep = parsed
    raison = valide(think, rep, attendu)
    if raison:
        return "", raison
    return f"Question : {q}\n<think>{think}</think>\nReponse : {rep}", ""


if __name__ == "__main__":
    main()
