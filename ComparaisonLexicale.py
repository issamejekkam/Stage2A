from database import database
import sys
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



KEYWORDS = {
    "Formation de base":          ["AFP", "CFC", "Bachelor", "Master", "doctorat", "université", "HES", "ES"],
    "Formation complémentaire":   ["complémentaire", "brevet", "CAS", "DAS", "MAS", "formations continues"],
    # "Durée":                      ["0", "1", "2", "3", "4", "5", "6", "7", "8", "ans", "année"],
    # "Nature":                     ["expérience", "préalable", "même type"],
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


nlp = spacy.load("fr_core_news_md")  

database= database("data.db")
database.connect()
if len(sys.argv) > 1 :
    CahierChargeName = sys.argv[1]
else:
    print("Usage: python evaluateFunction.py <CahierCharge>")
    sys.exit(1)

resultats=database.read_json(f"all_matches_of_{CahierChargeName}.json")



import re

def contains_keywords(text, text2, keywords, title):
    text = text.lower()
    text2 = text2.lower()
    if title in keywords:
        for keyword in keywords[title]:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text) and re.search(pattern, text2):
                print(f"Found keyword '{keyword}' in title '{title}'")
                return True
    return False
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def to_lower(text):
    return text.lower()

def tfidf_similarity(text1, text2):
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([text1, text2])

    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


resultats["sentence_lemmatized"] = resultats["sentence"].apply(to_lower)
resultats["sentence_lemmatized"] = resultats["sentence_lemmatized"].apply(lemmatize)
resultats['question_lemmatized'] = resultats['question'].apply(to_lower)
resultats['question_lemmatized'] = resultats['question_lemmatized'].apply(lemmatize)






similarities = []
for idx, row in resultats.iterrows():
    sim = tfidf_similarity(row['sentence_lemmatized'], row['question_lemmatized'])
    if contains_keywords(row['sentence'], row['question'], KEYWORDS, row['title']):
        sim += 1
    similarities.append(sim)
resultats['similarity'] = similarities
resultats=resultats.sort_values(by='similarity', ascending=False)
resultats_needed = resultats[["title", "question", "sentence", "score", "pts", "similarity"]]
resultats_needed.sort_values(by='similarity', ascending=False, inplace=True)


resultats_fin = resultats_needed.sort_values('similarity', ascending=False).drop_duplicates(subset=['title'], keep='first')


resultats_fin.to_json(f"results/after_comparaison_lexicale_of_{CahierChargeName}.json", orient="records", force_ascii=False, indent=4)

json_content = resultats_fin.to_json(orient="records", force_ascii=False, indent=4)
database.execute_query('''
    INSERT or replace INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
''', (f"after_comparaison_lexicale_of_{CahierChargeName}.json", json_content))

database.commit()
database.close()