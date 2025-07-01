import os
import pandas as pd
import sqlite3
from docx import Document
import sys
from io import BytesIO

from pretraitement import Pretraitement
from database import database
from similarity import Similarity
import json
import spacy
from sentence_transformers import SentenceTransformer, losses,InputExample

def main(docx_file=None):
    conn=database("data.db")
    conn.connect()
    CahierChargeName = docx_file

    df = conn.readQuestionnaire()
    CahierCharge = conn.readCahierDeCharges(CahierChargeName)
    questionnaire = df


    preproc = Pretraitement(CahierCharge, questionnaire)
    preproc._load_spacy()
    df= preproc.load_questionnaire()
    df=preproc.keep_abcd_lines(df)

    sentences = preproc.build_cahier_df()


    sim       = Similarity(batch_size=32)  
    # sim.train()
    # sim = Similarity(model_name="models/camembert_mnr_v1",
    #                  batch_size=32)   

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





    df_sentencized = df_sentencized[df_sentencized["sentence"].apply(contains_verb)]

    df_without_verbs = df_sentencized[~df_sentencized["sentence"].apply(contains_verb)]

        

    mappingSentencized = {}
    for question_title, sentences_list in mapping.items():
        sentencized = preproc.sentencize_sentences(sentences_list)
        mappingSentencized[question_title] = sentencized

    rows = []
    for title, sentences in mappingSentencized.items():
        for sentence in sentences:
            rows.append({'title': title, 'sentence': sentence})
    df_sentencized = pd.DataFrame(rows)


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

    banned=set()
    # Read a JSON file (example: 'input.json')
    with open('Triplets.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        banned.update(item["negatives"])



    for title in titles:
        questions = df[df["title"] == title]["response"].tolist()

        corpus_sentences = mappingSentencized.get(title, [])
        corpus_sentences = [s for s in corpus_sentences if s not in banned]
        if questions and corpus_sentences:
            matches = sim.top_k_matches(
                questions=questions,
                corpus_sentences=corpus_sentences,
                k=5,
                question_titles=[title]*len(questions),
            )
            all_matches.append(matches)

    matchesSentences = pd.concat(all_matches, ignore_index=True)


    import json


    sentences_used=[]
    results_for_json = []
    for title, group in matchesSentences.groupby('question_title'):
        max_score = group['score'].max()
        threshold = 0.1
        close_matches = group[(group['score'] >= max_score - threshold) & (group['score'] <= max_score + threshold)].sort_values('score', ascending=False)
        close_matches = close_matches.drop_duplicates(subset=["sentence"])
        # print(f"\n--- {title} (top score: {max_score:.3f}) ---")
        


        for _, row in close_matches.iterrows():
            if row['sentence'] not in sentences_used:
                sentences_used.append(row['sentence'])
            pts_value = df.loc[df["response"] == row["question"], "pts"]

            # print(f"Score: {row['score']:.3f}\nQuestion: {row['question']}\nSentence: {row['sentence']}\n")
            # print(f"pts: {pts_value.values[0]}")
            # print("-" * 80)

            results_for_json.append({
                "title": title,
                "question": row["question"],
                "sentence": row["sentence"],
                "score": row["score"],
                "pts": pts_value.values[0] 
            })



    if title == titles[-1]: 
        with open(f"results/results_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        json_content_str = json.dumps(results_for_json, ensure_ascii=False)
        conn.execute_query('''delete from ResultatsJSON where filename = ?''', (f"all_matches_of_{CahierChargeName}.json",))
        conn.execute_query('''
        INSERT INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
    ''', (f"all_matches_of_{CahierChargeName}.json", json_content_str))
        conn.commit()



    sentences = []
    for i in mappingSentencized:
        for j in mappingSentencized[i]:
            if j not in sentences:
                sentences.append(j)
    # print("Total unique sentences:", len(sentences))
    # print("\n",sentences)
    # print("Total sentences used:", len(sentences_used))
    # print("\n",sentences_used)


    to_use = [s for s in sentences if s not in sentences_used]
    to_use = to_use + df_without_verbs["sentence"].tolist()
    # print("Total sentences not used:", len(to_use))
    # for s in to_use:
    #     if len(s) > 2:
    #         print(s)

    with open(f"results/not_matched_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
        json.dump(to_use, f, ensure_ascii=False, indent=2)
    json_content_str = json.dumps(to_use, ensure_ascii=False)

    conn.execute_query('''delete from ResultatsJSON where filename = ?''', (f"not_matched_of_{CahierChargeName}.json",))

    conn.execute_query('''
        INSERT INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
    ''', (f"not_matched_of_{CahierChargeName}.json", json_content_str))

    conn.commit()



    conn.close()

def contains_verb(text):
    nlp = spacy.load("fr_core_news_md")
    doc = nlp(text)
    return any(token.pos_ == "VERB" or token.pos_ == "AUX" for token in doc)