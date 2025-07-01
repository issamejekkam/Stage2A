# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import sqlite3
from io import BytesIO
from textwrap import dedent

import pandas as pd
import spacy
from docx import Document
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, losses, InputExample

from pretraitement import Pretraitement
from database      import database
from similarity    import Similarity

# ─────────────────────────────────────────────────────────────────────────────
# 2. CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
SPACY_MODEL = "fr_core_news_md"
DB_PATH     = "data.db"
MODEL_PATH  = "models/Phi-3-mini-4k-instruct-q4.gguf"
TRIPLETS    = "Triplets.json"

KEYWORDS = {
    "Formation de base":          ["AFP", "CFC", "Bachelor", "Master", "doctorat", "université", "HES", "ES"],
    "Formation complémentaire":   ["complémentaire", "brevet", "CAS", "DAS", "MAS", "formations continues"],
    "Durée":                      [str(i) for i in range(9)] + ["ans", "année"],
    "Nature":                     ["expérience", "préalable", "même type"],
    "Autonomie de décision":      ["consignes", "directives", "autonome", "approuve"],
    "Responsabilités budgétaires":["budget", "financier", "facturation", "paiements", "suivi"],
    #...
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────
def contains_verb(text):
    """Renvoie True si on détecte au moins un VERB ou AUX dans le texte."""
    doc = nlp(text)
    return any(token.pos_ in {"VERB", "AUX"} for token in doc)

def choose_best_pair(title: str, pairs: list[dict]) -> int:
    """
    Utilise un prompt riche en contraintes + mots-clés pour forcer la sélection correcte.
    Retourne l’indice (0-based) du meilleur couple question-réponse.
    """
    kw = ", ".join(KEYWORDS.get(title, []))
    prompt = dedent(f"""
    Tu es évaluateur·trice RH, spécialiste du référentiel ANMEA.

    TÂCHE :
      • Pour chaque couple question / phrase candidate, choisis la **SEULE** phrase qui répond
        parfaitement et explicitement à la question.
      • prends en meilleur choix le couple ayant comme phrases contenant les mots-clés importants suivants: {kw if kw else ''}.
      • Ignore les phrases très courtes du type ‘Formation professionnelle requise’,
        ‘Compétences requises’, etc.
      • Accorde davantage de poids aux phrases contenant des verbes d’action pertinents.
    
    LISTE DES COUPLES :
    """).strip() + "\n\n"

    for i, pr in enumerate(pairs, 1):
        prompt += f"{i}. QUESTION : {pr['question']}\n   PHRASE   : {pr['sentence']}\n\n"
    prompt += "Réponds uniquement avec le chiffre du meilleur choix."

    out   = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.1
    )
    reply = out["choices"][0]["message"]["content"].strip()
    print(f"→ LLM a reçu le prompt :\n{prompt}")
    print(f"→ LLM a répondu : '{reply}'")

    try:
        return int(reply) - 1
    except:
        return 0

# ─────────────────────────────────────────────────────────────────────────────
# 4. INITIALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
nlp = spacy.load(SPACY_MODEL)

conn = database(DB_PATH)
conn.connect()

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    verbose=False
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # 5.1. Récupérer l’argument
    CahierChargeName = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # 5.2. Charger données
    df          = conn.readQuestionnaire()
    CahierCharge= conn.readCahierDeCharges(CahierChargeName)
    questionnaire = df

    # 5.3. Prétraitement
    preproc = Pretraitement(CahierCharge, questionnaire)
    preproc._load_spacy()
    df_q = preproc.load_questionnaire()
    df_q = preproc.keep_abcd_lines(df_q)
    sentences = preproc.build_cahier_df()

    # 5.4. Top-k matches initial
    sim = Similarity(batch_size=32)
    matches = sim.top_k_matches(
        questions=df_q["response"].tolist(),
        corpus_sentences=sentences["sentence"].tolist(),
        k=5,
        question_titles=df_q["title"].tolist(),
    )
    matches = matches.drop_duplicates(subset=["question_title", "sentence"])
    to_drop = [c for c in ["score", "rank", "label"] if c in matches]
    if to_drop:
        matches.drop(columns=to_drop, inplace=True)

    # Map question_title → phrases
    mapping = {}
    for _, row in matches.iterrows():
        mapping.setdefault(row["question_title"], []).append(row["sentence"])

    rows = []
    for title, sents in mapping.items():
        for sent in sents:
            rows.append({"title": title, "sentence": sent})
    df_sentencized = pd.DataFrame(rows)

    # 5.5. Filtrer les phrases sans verbe
    df_sent_with_verb    = df_sentencized[df_sentencized["sentence"].apply(contains_verb)]
    df_sent_without_verb = df_sentencized[~df_sentencized["sentence"].apply(contains_verb)]

    # 5.6. Re-sentencize et relancer Similarity par thème
    mappingSentencized = {
        title: preproc.sentencize_sentences(sents)
        for title, sents in mapping.items()
    }

    # Bannissement des négatives du JSON
    with open(TRIPLETS, "r", encoding="utf-8") as f:
        data = json.load(f)
    banned = {neg for item in data for neg in item["negatives"]}

    all_matches = []
    for title in [
        "Analyse et synthèse","Autonomie de décision","Complexité de l'environnement",
        # … la liste complète de tes titres …
    ]:
        questions       = df_q[df_q["title"] == title]["response"].tolist()
        corpus_sentences= [s for s in mappingSentencized.get(title, []) if s not in banned]
        if questions and corpus_sentences:
            m = sim.top_k_matches(
                questions=questions,
                corpus_sentences=corpus_sentences,
                k=5,
                question_titles=[title]*len(questions),
            )
            all_matches.append(m)
    matchesSentences = pd.concat(all_matches, ignore_index=True)

    # 5.7. Sélection finale avec LLM
    results_for_json = []
    sentences_used   = []
    for title, group in matchesSentences.groupby("question_title"):
        max_score  = group["score"].max()
        threshold  = 0.1
        close = (
            group[(group["score"] >= max_score - threshold) & (group["score"] <= max_score + threshold)]
            .drop_duplicates(subset=["sentence"])
            .sort_values("score", ascending=False)
        )
        for _, row in close.iterrows():
            if row["sentence"] not in sentences_used:
                sentences_used.append(row["sentence"])
            pts_value = df_q.loc[df_q["response"] == row["question"], "pts"].values[0]
            results_for_json.append({
                "title":    title,
                "question": row["question"],
                "sentence": row["sentence"],
                "score":    row["score"],
                "pts":      pts_value
            })

    # Sauvegarde all_matches + BDD
    out_all = f"results/all_matches_of_{CahierChargeName}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_all, "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=2)
    conn.execute_query(
        "INSERT OR REPLACE INTO ResultatsJSON (filename, json_content) VALUES (?,?)",
        (f"all_matches_of_{CahierChargeName}.json", json.dumps(results_for_json, ensure_ascii=False))
    )
    conn.commit()
    # Not matched
    all_sentences = sorted({s for sents in mappingSentencized.values() for s in sents})
    to_use = [s for s in all_sentences if s not in sentences_used]
    out_not = f"results/not_matched_of_{CahierChargeName}.json"
    with open(out_not, "w", encoding="utf-8") as f:
        json.dump(to_use, f, ensure_ascii=False, indent=2)

    conn.execute_query(
        "INSERT OR REPLACE INTO ResultatsJSON (filename, json_content) VALUES (?,?)",
        (f"not_matched_of_{CahierChargeName}.json", json.dumps(to_use, ensure_ascii=False))
    )

    # 5.8. Choix final LLM
    df_matches = conn.read_json(f"all_matches_of_{CahierChargeName}.json")
    final_results = []
    for title in df_matches["title"].unique():
        group = df_matches[df_matches["title"] == title].nlargest(10, "score")
        pairs = group[["question", "sentence", "score", "pts"]].to_dict("records")
        idx   = choose_best_pair(title, pairs)
        final_results.append(pairs[idx])

    out_llm = f"results/after_llm_of_{CahierChargeName}.json"
    with open(out_llm, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    conn.execute_query(
        "INSERT OR REPLACE INTO ResultatsJSON (filename, json_content) VALUES (?,?)",
        (f"after_llm_of_{CahierChargeName}.json", json.dumps(final_results, ensure_ascii=False))
    )

    conn.commit()
    conn.close()
    print("✅ Traitement terminé — résultats dans", out_llm)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
