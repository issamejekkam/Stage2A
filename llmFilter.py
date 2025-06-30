from llama_cpp import Llama
import json
import pandas as pd

# Charger le modèle local
llm = Llama(
    model_path="models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=2048,
    verbose=False
)

def choose_best_pair(pairs: list[dict]) -> int:
    """
    Donne une liste de paires {'question': ..., 'sentence': ...}
    Retourne l’indice de la meilleure paire selon le LLM
    """
    prompt = """Tu es un expert en ressources humaines et en analyse de cahier des charges.

    Ta tâche est de lire chaque question, puis d’évaluer laquelle des phrases proposées y répond correctement de façon explicite, précise et complète.

    Attention :
    - Ne choisis pas une phrase simplement "proche" ou "vaguement liée".
    - Ne choisis qu’une phrase qui **contient l'information exigée par la question**.
    - Ignore les phrases trop générales, vagues ou hors sujet.
    - les phrases courtes pouvant exprimer des titres comme "formation professionnelle requises, experiences professionnelles, compétences requises" ne sont pas pertinentes, et doivent pas etre prise en consideration.
    - fais attention aux mots cles qui peuvent repondent le plus aux questions comme "afp,hes,cfc,complémentaire...
    - donne plus d'importance aux phrases qui contiennent des verbes
    Réponds uniquement avec **le numéro** de la phrase correcte.

    Voici les paires question - phrase :
    """
    for i, pair in enumerate(pairs):
        prompt += f"{i+1}. Question : {pair['question']}\n   Phrase : {pair['sentence']}\n\n"

    prompt += "Réponds uniquement avec le chiffre du choix le plus pertinent."

    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.1
    )
    reply = output["choices"][0]["message"]["content"].strip()
    print(f"→ LLM a répondu : {reply}")
    try:
        return int(reply) - 1
    except:
        return 0  # fallback en cas d’erreur

# Charger les résultats
df_matches = pd.read_json("results/all_matches.json")

# Liste des titres uniques
titles = df_matches["title"].unique()

# Résultats finaux
results = []

for title in titles:
    group = df_matches[df_matches["title"] == title]
    pairs = group[["question", "sentence", "score", "pts"]].to_dict("records")

    best_idx = choose_best_pair(pairs)
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
