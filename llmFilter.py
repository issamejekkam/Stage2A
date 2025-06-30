from llama_cpp import Llama
import json
import pandas as pd
from database import database

# Charger le modèle local
llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    verbose=False
)

from llama_cpp import Llama
from textwrap import dedent

llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    verbose=False
)

# Mots-clés discriminants par grand thème – à adapter librement
KEYWORDS = {
    "Formation de base":          ["AFP", "CFC", "Bachelor", "Master", "doctorat", "université", "HES", "ES"],
    "Formation complémentaire":   ["complémentaire", "brevet", "CAS", "DAS", "MAS", "formations continues"],
    "Durée":                      ["0", "1", "2", "3", "4", "5", "6", "7", "8", "ans", "année"],
    "Nature":                     ["expérience", "préalable", "même type"],
    "Autonomie de décision":      ["consignes", "directives", "autonome", "approuve"],
    "Responsabilités budgétaires":["budget", "financier", "facturation", "paiements", "suivi"],
    # "Responsabilités de planification à court terme":
    #                                 ["court terme", "plan", "optimisation", "procédure"],
    # "Responsabilités de planification à long terme":
    #                                 ["long terme", "prospective", "anticiper", "recherches", "études"],
    # "Impact externe des prestations":
    #                                 ["image", "représentatif", "conséquences", "tiers"],
    # "Impact interne des prestations":
    #                                 ["coûts", "bon fonctionnement", "irréversibles"],
    # "Connaissances linguistiques":["français", "langue", "bilingue", "trilingue"],
    # "Nature des communications internes":["communications", "échanges", "négociations", "décisions"],
    # "Nature des communications externes":["communications", "informer", "explication", "négociations"],
    # "Complexité de l'environnement":["difficultés", "adaptabilité", "flexibilité"],
    # "Evolution de l'environnement":["évolution", "processus", "rapide"],
    # "Diversité des missions":     ["missions", "tâches", "diverses"],
    # "Diversité et quantité des postes à gérer":
    #                                 ["poste", "gère", "grand nombre", "activités"],
    # "Rôle dans la gestion des ressources humaines":
    #                                 ["animation", "conduite", "recrutement", "formation"],
    # "Innovation":                 ["adapter", "créer", "innovatrice", "concepts"]
}

def choose_best_pair(title: str, pairs: list[dict]) -> int:
    """
    Utilise un prompt riche en contraintes + mots-clés pour forcer la sélection correcte.
    Retourne l’indice (0-based) du meilleur couple question-réponse.
    """
    # 🔑 Récupère les mots-clés pertinents, sinon liste vide
    kw = ", ".join(KEYWORDS.get(title, []))

    prompt = dedent(f"""
    Tu es évaluateur·trice RH, spécialiste du référentiel ANMEA.

    TÂCHE :
      • Pour chaque couple question / phrase candidate, choisis la **SEULE** phrase qui répond
        parfaitement et explicitement à la question.
      •  prends en meilleure choix le couple ayant comme phrases contenant les mots-clés importants suivants: {kw if kw else ''}. 
      • Ne retiens pas les phrases génériques, incomplètes ou hors-sujet.
      • Ignore les phrases très courtes du type ‘Formation professionnelle requise’,
        ‘Compétences requises’, etc.
      • Accorde davantage de poids :
          
          – aux phrases contenant des verbes d’action pertinents.
      • Ta réponse doit être **un nombre entier unique** correspondant au bon choix.

    LISTE DES COUPLES :
    """).strip() + "\n\n"

    for i, pr in enumerate(pairs, 1):
        prompt += f"{i}. QUESTION : {pr['question']}\n   PHRASE   : {pr['sentence']}\n\n"

    prompt += "Réponds uniquement avec le chiffre du meilleur choix."

    # Appel modèle
    out = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.1
    )
    reply = out["choices"][0]["message"]["content"].strip()
    print(f"→ LLM a reçu le prompt :\n{prompt}")  # logging

    print(f"→ LLM a répondu : '{reply}'")  # logging

    try:
        return int(reply) - 1            # conversion 1-based → 0-based
    except Exception:
        return 0                         # fallback sûr




data=database("data.db")
data.connect()
df_matches=data.read_json("all_matches.json")

# Filtrer les lignes dont la phrase contient "Formation et expériences professionnelles requises"
df_matches = df_matches[~df_matches["sentence"].str.contains("Formation et expériences professionnelles requises", case=False, na=False)]

df_matches = df_matches[~df_matches["sentence"].str.contains("Activités et responsabilités principales", case=False, na=False)]

df_matches = df_matches[~df_matches["sentence"].str.contains("Responsable hiérarchique direct", case=False, na=False)]


# Liste des titres uniques
titles = df_matches["title"].unique()

# Résultats finaux
results = []

for title in titles:
    group = df_matches[df_matches["title"] == title]
    MAX_PAIRS = 10
    group = group.sort_values("score", ascending=False).head(MAX_PAIRS)
    pairs = group[["question", "sentence", "score", "pts"]].to_dict("records")

    best_idx = choose_best_pair(title,pairs)
    if 0 <= best_idx < len(pairs):
        best = pairs[best_idx]
        results.append({
            "title": title,
            "question": best["question"],
            "sentence": best["sentence"],
            "score": best["score"],
            "pts": best["pts"]
        })

# Sauvegarde dans after_llm.json
with open("results/after_llm.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

    json_content_str = json.dumps(results, ensure_ascii=False)
    data.execute_query('''DELETE FROM ResultatsJSON WHERE filename = ?''', ("after_llm.json",))
    data.execute_query('''
    INSERT INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
''', ("after_llm.json", json_content_str))
    data.commit()
data.close()
