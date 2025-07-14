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
    CahierChargeName = sys.argv[1]
    posteid = sys.argv[2] if len(sys.argv) > 2 else None
    userid = sys.argv[3] if len(sys.argv) > 3 else None
    fonctionPoste = sys.argv[4] if len(sys.argv) > 4 else None
    type = sys.argv[5] if len(sys.argv) > 5 else None
    lexicale = sys.argv[6] if len(sys.argv) > 6 else None
else:
    print("Usage: python evaluateFunction.py <CahierCharge>")
    sys.exit(1)


# Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------



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



if title == titles[-1]: 
    with open(f"results/all_matches_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=2)
    json_content_str_matched = json.dumps(results_for_json, ensure_ascii=False)
    conn.execute_query('''
    INSERT or replace INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
''', (f"all_matches_of_{CahierChargeName}.json", json_content_str_matched))
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


conn.execute_query('''
    INSERT or replace INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
''', (f"not_matched_of_{CahierChargeName}.json", json_content_str))

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
    conn.execute_query('''
        INSERT or replace INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
    ''', (f"all_matches_best_of_{CahierChargeName}.json", json_content))

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
                            INSERT INTO questionreponseposte (
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
                        INSERT INTO evaluationscompposte (posteid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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
                        INSERT INTO questionreponsefonction (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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
                        INSERT INTO evaluationscompfct (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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

    KEYWORDS = {
        # 1. Formation & expérience
        "Formation de base": [
            # tronc de la version XXL
            "niveau scolaire obligatoire", "niveaux scolaires obligatoires","niveau scolaire","formation post grade",
            "AFP", "AFPs", "CFC", "CFCs", "CAP", "CAPs", "BEP", "BEPs","SAS","maturité académique","certificat de fin de secondaire"
            "baccalauréat", "baccalauréats","bac,"
            "BTS", "BTSs", "DUT", "DUTs",
            "licence", "licences",
            "Bachelor", "Bachelors",
            "Master", "Masters",
            "Mastère spécialisé", "Mastères spécialisés",
            "MBA", "MBAs",
            "Doctorat", "Doctorats", "PhD", "PhDs",
            "habilitation", "habilitations", "VAE", "VAEs",

            "BAC", "bac", "bac+1", "bac+2", "bac+3", "bac+4", "bac+5",


        ],

        "Formation complémentaire": [
            "certification", "certifications", "certifié", "certifiée","certificat", "certificats",
            "certifiés", "certifiées",
            "formation continue", "formations continues",
            "complémentaire", "complémentaires","Une formation complémentaire","formation complémentaire","formations complémentaires","Deux formations complémentaires",
            "MOOC", "MOOCs",
            "micro-credential", "micro-credentials",
            "label qualité", "labels qualité",
            "examen fédéral", "examens fédéraux",
            "équivalence diplôme", "équivalences diplômes",
            "Titre RNCP", "Titres RNCP",
            "CQP", "CQPs",
            # ajouts
            "perfectionnement", "se perfectionner", "perfectionné",
            "spécialisation","spécialisation", "spécialiser", "spécialisé",
            "certifier", "certifiera",
            "actualiser", "actualisé",
            "recycler", "recyclage", "mettre à jour"
        ],

        "Nature": [
            "peu ou pas d'expérience", "peu ou pas d'expériences","de l'expérience professionnelle","même type d'expérience","même niveau de responsabilités",
            "première expérience", "premières expériences",
            "stage significatif", "stages significatifs",
            "expérience sectorielle", "expériences sectorielles",
            "expérience managériale", "expériences managériales",
            "expertise métier", "expertises métiers",
            "polyvalence sectorielle", "polyvalences sectorielles",
            "expérience internationale", "expériences internationales",
            "expérience multi-sites", "expériences multi-sites",
            # ajouts
            "pratiquer", "pratique", "pratiqué",
            "exercer", "exercé", "expérimenter", "expérimenté",
            "parcours", "background", "antécédent"
        ],

        "Durée": ["durée minimum", "durée minimale", "durée minimale requise",
            "moins d’un an", "moins d'une année",
            "0-1 an", "0 à 1 an",
            "1-3 ans", "1 à 3 ans",
            "3-5 ans", "3 à 5 ans",
            "5-8 ans", "5 à 8 ans",
            "plus de 8 ans d'expérience ",
            "8-12 ans", "8 à 12 ans",
            "12 ans et plus",
            "junior", "juniors",
            "intermédiaire", "intermédiaires",
            "confirmé", "confirmée", "confirmés", "confirmées",
            "senior", "seniors", "sénior", "séniors",
            # ajouts
            "ancienneté", "longevité", "court", "courte",
            "long", "longue", "prolongé", "prolongée",
            "permanent", "permanente"
        ],

        # 2. Responsabilités & autonomie
        "Responsabilités de planification et de réalisation des activités à court terme": [
            "élaboration", "élaborations","élaborer", "élaboré", "élaborée", "élaborés", "élaborées",
            "planification", "planifications",
            "suivi de plans à court terme", "suivis de plans à court terme","mise en oeuvre de plans à court terme","mise en oeuvre de plans","planification à long terme","planification",
            
            "planning quotidien", "plannings quotidiens",
            "ordonnancement", "ordonnancements",
            "suivi opérationnel", "suivis opérationnels",
            "gestion du temps réel", "gestions du temps réel",
            "KPI", "KPIs", "K.P.I.", "K.P.I.s",
            "optimisation processus", "optimisations processus",
            "processus optimisation", "optimisation des processus",
            # ajouts
            "exécuter", "exécution", "coordonner", "coordination",
            "mettre en œuvre", "mise en œuvre", "implémenter", "implémentation"
        ],

        "Responsabilités de planification et de réalisation des activités à long terme": [
            "plan directeur", "plans directeurs",
            "roadmap stratégique", "roadmaps stratégiques",
            "roadmap", "roadmaps",
            "réalisations futures","recherches futures","rechereches",
            "budget pluriannuel", "budgets pluriannuels",
            "veille prospective", "veilles prospectives",
            "business plan", "business plans",
            "schéma directeur", "schémas directeurs",
            "plan d’investissement", "plans d’investissement",
            # ajouts
            "anticiper", "anticipation", "projeter", "projection",
            "concevoir", "conception", "définir", "définition",
            "élaborer", "élaboration", "visionner", "vision stratégique"
        ],

        "Autonomie de décision": [
            "applique directives", "appliqué", "appliquée", "appliqués", "appliquées",
            "propose améliorations", "proposé", "proposée", "proposés", "proposées",
            "décide", "décider", "décidé", "décidée", "décidés", "décidées",
            "budget délégué", "budgets délégués",
            "signature engageante", "signatures engageantes",
            "approbation finale", "approbations finales",
            "pleine latitude", "pleines latitudes",
            "pouvoir disciplinaire", "pouvoirs disciplinaires",
            # ajouts
            "décider", "décision", "trancher", "arbitrer", "arbitrage",
            "autoriser", "autorisation", "valider", "validation",
            "indépendant", "indépendante", "autonome", "autonomes",
            "consigne", "consignes","directive", "directives","plan de travail","contrôle d'un responsable"
        ],

        "Diversité et quantité des postes à gérer": [
            "gère directement peu de postes","peu de postes","activités semblables","peu de postes","activités diverses","grand nombre de postes",
            "aucun subordonné", "aucune subordonnée", "aucuns subordonnés",
            "gestion de proximité", "gestions de proximité",
            "management intermédiaire", "managements intermédiaires",
            "management transversal", "managements transversaux",
            "direction hiérarchique", "directions hiérarchiques",
            "leadership matriciel", "leaderships matriciels",
            "animation réseau", "animations réseau",
            "pilotage multi-équipes", "pilotages multi-équipes",
            # ajouts
            "encadrer", "encadrement", "superviser", "supervision",
            "diriger", "direction", "manager", "management",
            "mobiliser", "mobilisation", "effectif", "effectifs", "staff"
        ],

        "Rôle dans la gestion des ressources humaines": [
            "ressources humaines","transfert de connaissances","formation des collaborateurs","coordonne l'exécution","coordonne","coordonné", "coordonnée", "coordonnés", "coordonnées","coordonner",
            "évaluation des fonctions","appréciation des collaborateurs"
            "coaching", "coachings", "coach", "coachs",
            "direction", "directions","recrutement", "recrutements","rémunération"
            "gestion des ressources humaines", "gestions des ressources humaines",
            "animation d’équipe", "animations d’équipe",
            "recrutement", "recrutements",
            "onboarding", "onboardings",
            "évaluation des performances", "évaluations des performances",
            "GPEC", "G.P.E.C.",
            "gestion talents", "gestion de talents", "gestions de talents",
            "négociation sociale", "négociations sociales",
            # ajouts
            "motiver", "motivation", "coacher", "coaché",
            "développer", "développement", "accompagner", "accompagnement",
            "assessor", "discipliner", "discipline"
        ],

        "Responsabilités budgétaires": [
            "suivi financier", "suivis financiers","poste budgétaire","postes budgétaires",
            "gestion budget", "gestions budgets",
            "facturation", "facturations","paiement", "paiements",
            "gestion financière", "gestions financières",
            "gestion des coûts", "gestions des coûts",
            "gestion des dépenses", "gestions des dépenses",
            "gestion des factures", "gestions des factures",
            "traitement factures", "traitements factures",
            "suivi poste budgétaire", "suivis postes budgétaires",
            "gestion centre de coûts", "gestions centres de coûts",
            "contrôle de gestion", "contrôles de gestion",
            "élaboration budget", "élaborations budget",
            "arbitrage financier", "arbitrages financiers",
            "engagement dépenses", "engagements dépenses",
            "reporting financier", "reportings financiers",
            "analyse écarts", "analyses écarts",
            "closing comptable", "closings comptables",
            # ajouts
            "budget", "budgets", "financier", "financiers","budgetaire", "budgétaires",
            "financer", "financement", "budgéter", "budgétisation",
            "payer", "paiement", "investir", "investissement",
            "calculer", "calcul", "dépenser", "dépense"
        ],

        # 3. Impact
        "Impact externe des prestations": [
            "image de marque", "images de marque",
            "orientation client", "orientations client", "orientations clients",
            "satisfaction bénéficiaire", "satisfactions bénéficiaires",
            "impact sociétal", "impacts sociétaux",
            "responsabilité sociale", "responsabilités sociales",
            "relation publique", "relations publiques",
            "influence marché", "influences marchés",
            "engagement parties prenantes", "engagements parties prenantes",
            # ajouts
            "influencer", "influence", "promouvoir", "promotion",
            "représenter", "représentation", "défendre", "défense",
            "négocier", "négociation", "visibilité", "notoriété"
        ],

        "Impact interne des prestations": [
            "prestation de service", "prestations de services","prestation fournie","bénéficiare de la prestation","bénéficiare de prestations","bénéficiares de prestations",
            "réduction coûts", "réduction des coûts", "réductions des coûts",
            "qualité service", "qualité du service", "qualités service",
            "sécurité processus", "sécurités processus",
            "productivité", "productivités",
            "efficacité opérationnelle", "efficacités opérationnelles",
            "rentabilité", "rentabilités",
            "gestion changement", "gestion des changements", "gestions changement",
            "transformation digitale", "transformations digitales",
            # ajouts
            "optimiser", "optimisation", "améliorer", "amélioration",
            "réduire", "réduction", "sécuriser", "sécurisation",
            "stabiliser", "stabilisation", "harmoniser", "harmonisation",
            "standardiser", "standardisation", "rationaliser", "rationalisation"
        ],

        # 4. Communication & langues
        "Nature des communications internes": [
            "sujet courant", "sujets courants","l'échange d'informations","l'échange d'information","échanges d'informations",
            "briefing", "briefings","interactions"
            "reporting interne", "reportings internes",
            "animation réunion", "animations réunion",
            "synthèse", "synthèses",
            "négociation interne", "négociations internes","négociation", "négociations",
            "facilitation", "facilitations",
            "communication transverse", "communications transverses",
            "gestion de conflit", "gestions de conflits",
            "culture feedback", "cultures feedback",
            # ajouts
            "informer", "information", "échanger", "échange","négocier","négociateur",
            "communiquer", "communication", "dialoguer", "dialogue",
            "transmettre", "transmission", "expliquer", "explication",
            "coordonner", "coordination"
        ],

        "Nature des communications externes": [
            "relation client", "relations clients", "relations clientèle","sujet courant","sujets courants",
            "relations fournisseurs","explication","perceptions",
            "relations institutionnelles",
            "porte-parole", "porte-paroles",
            "représentation officielle", "représentations officielles",
            "communication de crise", "communications de crise",
            "relation presse", "relations presse",
            "lobbying",
            "community management", "community managements",
            "gestion partenariats", "gestions partenariats",
            # ajouts
            "informer", "information", "expliquer", "explication",
            "présenter", "présentation", "convaincre", "conviction",
            "persuader", "persuasion", "argumenter", "argumentation",
            "promouvoir", "promotion", "négocier", "négociation"
        ],

        "Connaissances linguistiques": [
            "A1", "A2", "B1", "B2", "C1", "C2",
            "français courant", "français courants",
            "anglais professionnel", "anglais professionnels",
            "allemand technique", "allemands techniques",
            "espagnol opérationnel", "espagnols opérationnels",
            "italien conversationnel", "italiens conversationnels",
            "langue des signes", "langues des signes","langues","langue",
            "bilingue", "bilingues","deux langues","deuxième langue","troisième langue",
            "trilingue", "trilingues",
            "multilingue", "multilingues","lecture","coversation","rédaction","français","française","langue maternelle",
            # ajouts
            "parler", "parlé", "lire", "lu", "rédiger", "rédigé","écrire ","écrit",
            "traduire", "traduction", "interpréter", "interprétation",
            "fluency", "maîtrise", "maîtriser", "maîtrisé"
        ],

        # 5. Environnement & problèmes
        "Complexité de l'environnement": [
            "procédure standard", "procédures standard", "procédures standards",
            "environnement régulé", "environnements régulés",
            "incertitude forte", "incertitudes fortes",
            "contexte VUCA", "contextes VUCA",
            "multi-site", "multi-sites", "multisite", "multisites",
            "multi-projet", "multi-projets", "multiprojet", "multiprojets",
            "environnement à risque", "environnements à risque",
            "cadre international", "cadres internationaux",
            "difficultés","adaptabilité", "flexibilité",
            "complexité", "complexités",
            # ajouts
            "gérer", "gestion", "faire face", "affronter",
            "naviguer", "navigation", "complexifier", "complexification",
            "s'adapter", "adaptation", "volatil", "incertain", "ambigu", "turbulent"
        ],

        "Evolution de l'environnement": [
            "évolution régulière","adaptation","évolution rapide","évolution très rapide","processus de travail"
            "environnement à évolution régulière", "environnements à évolution régulière",
            "environnement à évolution rapide", "environnements à évolution rapide",
            "environnement à évolution très rapide", "environnements à évolution très rapide",
            "adaptation processus", "adaptations processus",
            "transformation des pratiques", "transformations des pratiques",
            # ajouts
            "évoluer", "évolution", "changer", "changement",
            "transformer", "transformation", "adapter", "adaptation",
            "progresser", "progression", "métamorphoser", "métamorphose"
        ],

        "Diversité des missions": [
            "tâches principales","même mission","2 à 3 missions différentes","issions différentes","4 missions différentes"
            "tâche répétitive", "tâches répétitives",
            "mission variée", "missions variées",
            "poly-compétence", "poly-compétences", "polycompétence", "polycompétences",
            "projet transverse", "projets transverses",
            "portefeuille diversifié", "portefeuilles diversifiés",
            "multi-business-unit", "multi-business-units", "multi BU", "multi BUs",
            # ajouts
            "diversifier", "diversification", "varier", "variation",
            "polyvalent", "polyvalente", "multitâche", "multitâches",
            "élargir", "élargissement", "alternatif", "alternative"
        ],

        "Analyse et synthèse": [
            "base d'analyse", "bases d'analyses",
            "analyse de données", "analyses de données",
            "analyse statistique", "analyses statistiques",
            "analyse de marché", "analyses de marché",
            "analyse de performance", "analyses de performance",
            "analyse de risque", "analyses de risque",
            "analyse de tendance", "analyses de tendance",
            "analyse de processus", "analyses de processus",
            "analyse de performance", "analyses de performance",
            "analyse de la concurrence", "analyses de la concurrence",
            "analyse de l'impact", "analyses de l'impact",
            "analyse de la satisfaction", "analyses de la satisfaction",
            "analyse de la qualité", "analyses de la qualité",
            "analyse de la rentabilité", "analyses de la rentabilité",
            "analyse de la productivité", "analyses de la productivité",
            "analyse de la chaîne de valeur", "analyses de la chaîne de valeur",
            "interprétation", "interprétations","interpréter","synthèse", "synthèses",
            "analyse de la supply chain", "analyses de la supply chain","sujets parfois complexes"
            "analyse", "analyses", "analyser", "analysé", "analysée", "analysés", "analysées",
            "collecte donnée", "collecte données", "collectes de données",
            "data-crunching", "data crunching", "data-crunchings",
            "diagnostic", "diagnostics",
            "analyse", "analyses", "analyser", "analysé", "analysée", "analysés", "analysées",
            "analytique", "analytiques",
            "business intelligence", "BI",
            "tableau de bord", "tableaux de bord",
            "modélisation", "modélisations", "modéliser", "modélisé", "modélisée", "modélisés", "modélisées",
            "recommandation stratégique", "recommandations stratégiques",
            # ajouts
            "évaluer", "évaluation", "interpréter", "interprétation",
            "conclure", "conclusion", "résumer", "résumé",
            "décomposer", "décomposition", "diagnostiquer", "auditer", "audit"
        ],

        "Innovation": [
            "solutions", "solution", "solutions innovantes", "solution innovante",
            "innovation", "innovations", "innovateur", "innovatrice",
            "innovation produit", "innovations produits",
            "amélioration continue", "améliorations continues",
            "lean management", "lean",
            "design thinking",
            "idéation", "idéations",
            "proof of concept", "proofs of concept",
            "prototype", "prototypes", "prototypage", "prototypages",
            "R&D", "recherche et développement",
            "gestion innovation", "gestions innovation",
            "open innovation", "open innovations",
            "veille technologique", "veilles technologiques",
            # ajouts
            "créer", "création", "inventer", "invention",
            "concevoir", "imagination", "imaginer",
            "réinventer", "réinvention", "repenser",
            "disrupter", "disruption"
        ]
    }
    resultats=pd.DataFrame(json_content_str_matched)
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
    conn.execute_query('''
        INSERT or replace INTO ResultatsJSON (filename, json_content) VALUES (?, ?)
    ''', (f"after_comparaison_lexicale_of_{CahierChargeName}.json", json_content))

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
                            INSERT INTO questionreponseposte (
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
                        INSERT INTO evaluationscompposte (posteid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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
                        INSERT INTO questionreponsefonction (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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
                        INSERT INTO evaluationscompfct (fctid, evalid, critereid, sscritereid, questionnombre, Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid)
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