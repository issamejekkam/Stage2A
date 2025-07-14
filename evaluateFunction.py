# Import necessary libraries
#--------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import sqlite3
import sys
import json
import spacy


# Import Classes
#--------------------------------------------------------------------------------------------------------------------------------------------------


from pretraitement import Pretraitement
from database import database
from similarity import Similarity

# load spacy model & database
#--------------------------------------------------------------------------------------------------------------------------------------------------


nlp = spacy.load("fr_core_news_md")  
conn=database("data.db")
conn.connect()


# store parameters from command line arguments
#--------------------------------------------------------------------------------------------------------------------------------------------------


if len(sys.argv) > 1 :
    if len(sys.argv) > 1:
        CahierChargeName = sys.argv[1]
    else:
        print("Error: CahierChargeName argument is missing.")
        sys.exit(1)
    if len(sys.argv) > 2:
        posteid = sys.argv[2]
    else:
        print("Error: posteid argument is missing.")
        sys.exit(1)

    if len(sys.argv) > 3:
        userid = sys.argv[3]
    else:
        print("Error: userid argument is missing.")
        sys.exit(1)

    if len(sys.argv) > 4:
        fonctionPoste = sys.argv[4]
    else:
        print("Error: fonctionPoste argument is missing.")
        sys.exit(1)

    if len(sys.argv) > 5:
        type = sys.argv[5]
    else:
        print("Error: type argument is missing.")
        sys.exit(1)

    lexicale = sys.argv[6] if len(sys.argv) > 6 else None
else:
    print("Usage: python evaluateFunction.py <CahierCharge>")
    sys.exit(1)


# Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def contains_verb(text):
    doc = nlp(text)
    return any(token.pos_ == "VERB" or token.pos_ == "AUX" for token in doc)


def twowordsmin(text):
    return len(text.split()) >= 2


# Pretraitement of documents
#--------------------------------------------------------------------------------------------------------------------------------------------------


df = conn.readQuestionnaire()
CahierCharge = conn.readCahierDeCharges(CahierChargeName)
questionnaire = df

preproc = Pretraitement(CahierCharge, questionnaire)
preproc._load_spacy()
df= preproc.load_questionnaire()
df=preproc.keep_abcd_lines(df)
sentences = preproc.build_cahier_df()


# Catch the five best matches for each long sentence (sentencization with spacy)
#--------------------------------------------------------------------------------------------------------------------------------------------------


sim       = Similarity(batch_size=32)  
matches   = sim.top_k_matches(
    questions=df["response"].tolist(),
    corpus_sentences=sentences["sentence"].tolist(),
    k=5,
    question_titles=df["title"].tolist(),
)


matches_no_duplicates = matches.drop_duplicates(subset=["question_title", "sentence"])
cols_to_drop = [col for col in ["score", "rank", "label"] if col in matches_no_duplicates.columns]
if cols_to_drop:
    matches_no_duplicates.drop(columns=cols_to_drop, inplace=True)

mapping={}
for i, row in matches_no_duplicates.iterrows():
    question_title = row["question_title"]
    sentence = row["sentence"]
    if question_title not in mapping:
        mapping[question_title] = []
    mapping[question_title].append(sentence)

rows = []
for title, sentences in mapping.items():
    for sentence in sentences:
        rows.append({'title': title, 'sentence': sentence})
df_sentencized = pd.DataFrame(rows)


# Catch the five best matches for each  sentence (sentencization with punctuation)
#--------------------------------------------------------------------------------------------------------------------------------------------------



mappingSentencized = {}
for question_title, sentences_list in mapping.items():
    sentencized = preproc.sentencize_sentences(sentences_list)
    mappingSentencized[question_title] = sentencized

rows = []
for title, sentences in mappingSentencized.items():
    for sentence in sentences:
        rows.append({'title': title, 'sentence': sentence})
df_sentencized = pd.DataFrame(rows)
# df_sentencized = df_sentencized[df_sentencized["sentence"].apply(contains_verb)]
df_sentencized = df_sentencized[df_sentencized["sentence"].apply(twowordsmin)]

# df_without_verbs = df_sentencized[~df_sentencized["sentence"].apply(contains_verb)]
df_without_verbs = df_sentencized[~df_sentencized["sentence"].apply(twowordsmin)]

for title in mappingSentencized:
    # mappingSentencized[title] = [s for s in mappingSentencized[title] if contains_verb(s)]
    mappingSentencized[title] = [s for s in mappingSentencized[title] if twowordsmin(s)]

sim       = Similarity(batch_size=32)   
titles = [
    "Analyse et synthèse",
    "Autonomie de décision",
    "Complexité de l'environnement",
    "Connaissances linguistiques",
    "Diversité des missions",
    "Diversité et quantité des postes à gérer",
    "Durée",
    "Evolution de l'environnement",
    "Formation complémentaire",
    "Formation de base",
    "Impact externe des prestations",
    "Impact interne des prestations",
    "Innovation",
    "Nature",
    "Nature des communications externes",
    "Nature des communications internes",
    "Responsabilités budgétaires",
    "Responsabilités de planification et de réalisation des activités à court terme",
    "Responsabilités de planification et de réalisation des activités à long terme",
    "Rôle dans la gestion des ressources humaines"
]
all_matches = []





for title in titles:
    questions = df[df["title"] == title]["response"].tolist()

    corpus_sentences = mappingSentencized.get(title, [])
    if questions and corpus_sentences:
        matches = sim.top_k_matches(
            questions=questions,
            corpus_sentences=corpus_sentences,
            k=5,
            question_titles=[title]*len(questions),
        )
        all_matches.append(matches)

matchesSentences = pd.concat(all_matches, ignore_index=True)



# store the list of responses to all_matches and not_matched
#--------------------------------------------------------------------------------------------------------------------------------------------------


sentences_used=[]
results_for_json = []
for title, group in matchesSentences.groupby('question_title'):
    max_score = group['score'].max()
    threshold = 0.3
    close_matches = group[(group['score'] >= max_score - threshold) & (group['score'] <= max_score + threshold)].sort_values('score', ascending=False)
    close_matches = close_matches.drop_duplicates(subset=["sentence"])
    


    for _, row in close_matches.iterrows():
        if row['sentence'] not in sentences_used:
            sentences_used.append(row['sentence'])
        pts_value = df.loc[df["response"] == row["question"], "pts"]

        results_for_json.append({
            "title": title,
            "question": row["question"],
            "sentence": row["sentence"],
            "score": row["score"],
            "pts": pts_value.values[0] 
        })


results_for_json_to_use_for_semantique=results_for_json
if title == titles[-1]: 
    with open(f"results/all_matches_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=2)
    json_content_str_matched = json.dumps(results_for_json, ensure_ascii=False)
    jsoncontentdf= pd.DataFrame(results_for_json)

    for row in jsoncontentdf.itertuples(index=False):
        conn.execute_query(
            '''
             INSERT OR REPLACE INTO all_matches (
                filename,title, question, sentence, score, pts
            ) VALUES (?,?, ?,?,?, ?)
            ''',
            (
                f"{CahierChargeName}",
                row.title,
                row.question,
                row.sentence,
                row.score,
                row.pts
            ))





    conn.commit()


sentences = []
for i in mappingSentencized:
    for j in mappingSentencized[i]:
        if j not in sentences:
            sentences.append(j)

to_use = [s for s in sentences if s not in sentences_used]

with open(f"results/not_matched_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
    json.dump(to_use, f, ensure_ascii=False, indent=2)
json_content_str = json.dumps(to_use, ensure_ascii=False)


for row in to_use:
    conn.execute_query('''
        INSERT or replace INTO not_matched (filename, sentence) VALUES (?, ?)
    ''', (f"{CahierChargeName}", row))


conn.commit()
results_for_json = pd.DataFrame(results_for_json)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
# -------------------------semantique----------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


if lexicale == "false":

    results_for_json = results_for_json.sort_values(by='score', ascending=False)
    results_for_json = results_for_json.drop_duplicates(subset=['title'], keep='first')  
    questionnaireClean = conn.readQuestionnaireClean()
    for i in results_for_json.index:
        for index, row in questionnaireClean.iterrows():
            if 'codeReponse' not in results_for_json.columns:
                results_for_json.insert(3, 'codeReponse', '') 
            if 'codeQuestion' not in results_for_json.columns:
                results_for_json.insert(1, 'codeQuestion', '') 
            

            if row["reponsedesc"] == results_for_json.at[i, 'question']:
                    a = row['critereid']
                    b = row['sscritereid']
                    c = row['questionnombre']
                    d = row['reponse']
                    results_for_json.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                    results_for_json.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"
    results_for_json=results_for_json.rename(columns={
        'title': 'Question',
        'question': 'niv de réponse'})
    results_for_json.to_json(f"results/all_matches_best_of_{CahierChargeName}.docx.json", orient="records", force_ascii=False, indent=4)


    json_content = results_for_json.to_json(orient="records", force_ascii=False, indent=4)

    conn.commit()
    resultats_finale=results_for_json['codeReponse']

    resultats_finale_sorted= resultats_finale.sort_values(ascending=True).reset_index(drop=True)
    resultats_finale_sorted


    temp = resultats_finale_sorted.str.split('-', expand=True)

    # Étape 2 : séparer la partie gauche du tiret (avant le -) par '.'
    left_parts = temp[1].str.split('.', expand=True)

    toStored = pd.DataFrame({
        'posteid': posteid,
        'evalid': 200,
        'critereid': temp[0],
        'sscritereid': left_parts[0],
        'questionnombre': left_parts[1],
        'Reponsenivmin': left_parts[2],
        'Reponsenivmax': left_parts[2],
        'desactive': 0,
        'lastupdated': None,
        'usrid': userid})

    if fonctionPoste is not None:
        if fonctionPoste=="poste":
            if type is not None:
                if type=="responsabilités":
                    for row in toStored.itertuples(index=False):
                        conn.execute_query(
                            '''
                            INSERT or replace INTO questionreponseposte (
                                posteid, evalid, critereid, sscritereid, questionnombre,
                                Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid
                            )
                        )
                elif type=="compétences":
                    for row in toStored.itertuples(index=False):
                        
                        conn.execute_query('''
                        INSERT or replace INTO evaluationscompposte (posteid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
        if fonctionPoste=="fonction":
            if type is not None:
                if type=="responsabilités":
                    for row in toStored.itertuples(index=False):

                        conn.execute_query('''
                        INSERT or replace INTO questionreponsefonction (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
                elif type=="compétences":
                    for row in toStored.itertuples(index=False):
                        conn.execute_query('''
                        INSERT or replace INTO evaluationscompfct (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
    conn.commit()


#------------------------------------------------------------------------
#------------------------------------------------------------------------
# -------------------------lexicale----------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------



if lexicale=="true":

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import re

    KEYWORDS = conn.readkeywords()
    resultats=pd.DataFrame(results_for_json_to_use_for_semantique)
    def contains_keywords(text, text2, keywords, title):
        text = text.lower()
        if text2 is not None:
            text2 = text2.lower()
        if title in keywords:
            for keyword in keywords[title]:
                pattern = r'\b' + re.escape(keyword.lower()) + r's?\b'
                if re.search(pattern, text) and (re.search(pattern, text2) if text2 is not None else True):
                    print(f"Found keyword '{keyword}' in title '{title}'")
                    return True
        return False


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
        elif contains_keywords(row['sentence'],None, KEYWORDS, row['title']):
            sim += 0.5
        similarities.append(sim)
    resultats['similarity'] = similarities
    resultats=resultats.sort_values(by='similarity', ascending=False)
    resultats_needed = resultats[["title", "question", "sentence", "score", "pts", "similarity"]]
    resultats_needed.sort_values(by='similarity', ascending=False, inplace=True)

    resultats_fin = resultats_needed.sort_values('similarity', ascending=False).drop_duplicates(subset=['title'], keep='first')
    questionnaireClean = conn.readQuestionnaireClean()
    for i in resultats_fin.index:
        for index, row in questionnaireClean.iterrows():
            if 'codeReponse' not in resultats_fin.columns:
                resultats_fin.insert(3, 'codeReponse', '') 
            if 'codeQuestion' not in resultats_fin.columns:
                resultats_fin.insert(1, 'codeQuestion', '') 
            

            if row["reponsedesc"] == resultats_fin.at[i, 'question']:
                    a = row['critereid']
                    b = row['sscritereid']
                    c = row['questionnombre']
                    d = row['reponse']
                    resultats_fin.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                    resultats_fin.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"
    resultats_fin=resultats_fin.rename(columns={
        'title': 'Question',
        'question': 'niv de réponse'})



    resultats_fin.to_json(f"results/after_comparaison_lexicale_of_{CahierChargeName}.json", orient="records", force_ascii=False, indent=4)

    json_content = resultats_fin.to_json(orient="records", force_ascii=False, indent=4)
 
    conn.commit()
    resultats_finale=resultats_fin['codeReponse']

    resultats_finale_sorted= resultats_finale.sort_values(ascending=True).reset_index(drop=True)
    resultats_finale_sorted


    temp = resultats_finale_sorted.str.split('-', expand=True)

    # Étape 2 : séparer la partie gauche du tiret (avant le -) par '.'
    left_parts = temp[1].str.split('.', expand=True)

    toStored = pd.DataFrame({
        'posteid': posteid,
        'evalid': 200,
        'critereid': temp[0],
        'sscritereid': left_parts[0],
        'questionnombre': left_parts[1],
        'Reponsenivmin': left_parts[2],
        'Reponsenivmax': left_parts[2],
        'desactive': 0,
        'lastupdated': None,
        'usrid': userid})

    if fonctionPoste is not None:
        if fonctionPoste=="poste":
            if type is not None:
                if type=="responsabilités":
                    for row in toStored.itertuples(index=False):
                        conn.execute_query(
                            '''
                            INSERT or replace INTO questionreponseposte (
                                posteid, evalid, critereid, sscritereid, questionnombre,
                                Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''',
                            (
                                row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid
                            )
                        )
                elif type=="compétences":
                    for row in toStored.itertuples(index=False):
                        
                        conn.execute_query('''
                        INSERT or replace INTO evaluationscompposte (posteid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
        if fonctionPoste=="fonction":
            if type is not None:
                if type=="responsabilités":
                    for row in toStored.itertuples(index=False):

                        conn.execute_query('''
                        INSERT or replace INTO questionreponsefonction (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
                elif type=="compétences":
                    for row in toStored.itertuples(index=False):
                        conn.execute_query('''
                        INSERT or replace INTO evaluationscompfct (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',         ( row.posteid,
                                row.evalid,
                                row.critereid,
                                row.sscritereid,
                                row.questionnombre,
                                row.Reponsenivmin,
                                row.Reponsenivmax,
                                row.desactive,
                                row.lastupdated,
                                row.usrid))
    conn.commit()



 
conn.close()