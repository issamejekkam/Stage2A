from database import database
import sys

database= database("data.db")
database.connect()
if len(sys.argv) > 1 :
    CahierChargeName = sys.argv[1]
else:
    print("Usage: python evaluateFunction.py <CahierCharge>")
    sys.exit(1)

resultats=database.read_json(f"all_matches_of_{CahierChargeName}.json")


import spacy
nlp = spacy.load("fr_core_news_md")  # pas besoin d'exclude

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def to_lower(text):
    return text.lower()


resultats["sentence_lemmatized"] = resultats["sentence"].apply(to_lower)
resultats["sentence_lemmatized"] = resultats["sentence_lemmatized"].apply(lemmatize)
resultats['question_lemmatized'] = resultats['question'].apply(to_lower)
resultats['question_lemmatized'] = resultats['question_lemmatized'].apply(lemmatize)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def tfidf_similarity(text1, text2):
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([text1, text2])

    return cosine_similarity(tfidf[0], tfidf[1])[0][0]


resultats['similarity'] = resultats.apply(lambda row: tfidf_similarity(row['sentence_lemmatized'], row['question_lemmatized']), axis=1)
resultats=resultats.sort_values(by='similarity', ascending=False)
resultats_needed = resultats[["title", "question", "sentence", "score", "pts", "similarity"]]


resultats_fin = resultats_needed.sort_values('similarity', ascending=False).drop_duplicates(subset=['title'], keep='first')


resultats_fin.to_json(f"results/after_comparaison_lexicale_of_{CahierChargeName}.json", orient="records", force_ascii=False, indent=4)