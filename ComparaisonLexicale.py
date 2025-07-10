from database import database
import sys
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KEYWORDS = {
    "Formation de base":          ["AFP", "CFC", "Bachelor", "Master", "doctorat", "université", "HES", "ES"],
    "Formation complémentaire":   ["complémentaire","complétée","certification"],
    "Durée":                      [ "ans", "année"],
    "Nature":                     ["professionnelle","expérience", "préalable", "même type"],
    "Autonomie de décision":      ["consignes", "directives", "autonome", "autonomie","approuve"],
    "Responsabilités budgétaires":["budget", "financier", "facturation", "paiements"],
    # "Responsabilités de planification à court terme":
    #                                 ["court terme", "plan", "optimisation", "procédure"],
    # "Responsabilités de planification à long terme":
    #                                 ["long terme", "prospective", "anticiper", "recherches", "études"],
    # "Impact externe des prestations":
    #                                 ["image", "représentatif", "conséquences", "tiers"],
    # "Impact interne des prestations":
    #                                 ["coûts", "bon fonctionnement", "irréversibles"],
    "Connaissances linguistiques":["français","française", "langue", "bilingue", "trilingue"],
    "Nature des communications internes":["communications", "échanges", "négociations", "décisions"],
    "Nature des communications externes":["communications", "informer", "explication", "négociations"],
    "Complexité de l'environnement":["difficultés", "adaptabilité", "flexibilité","complexe"],
    # "Evolution de l'environnement":["évolution", "processus", "rapide"],
    "Diversité des missions":     ["missions", "tâches", "diverses"],
    # "Diversité et quantité des postes à gérer":
    #                                 ["poste", "gère", "grand nombre", "activités"],
    "Rôle dans la gestion des ressources humaines":
                                    ["animation", "recrutement", "humaines", "ressources humaines"],
    "Innovation":                 ["innovation", "innovatrice"]
}

KEYWORDS = {
    # 1. Formation & expérience
    "Formation de base": [
        # tronc de la version XXL
        "niveau scolaire obligatoire", "niveaux scolaires obligatoires",
        "AFP", "AFPs", "CFC", "CFCs", "CAP", "CAPs", "BEP", "BEPs",
        "baccalauréat", "baccalauréats",
        "BTS", "BTSs", "DUT", "DUTs",
        "licence", "licences",
        "Bachelor", "Bachelors",
        "Master", "Masters",
        "Mastère spécialisé", "Mastères spécialisés",
        "MBA", "MBAs",
        "Doctorat", "Doctorats", "PhD", "PhDs",
        "habilitation", "habilitations", "VAE", "VAEs",
        # ajouts « forts »
        "BAC", "bac", "bac+1", "bac+2", "bac+3", "bac+4", "bac+5",


    ],

    "Formation complémentaire": [
        "certification", "certifications", "certifié", "certifiée",
        "certifiés", "certifiées",
        "formation continue", "formations continues",

        "MOOC", "MOOCs",
        "micro-credential", "micro-credentials",
        "label qualité", "labels qualité",
        "examen fédéral", "examens fédéraux",
        "équivalence diplôme", "équivalences diplômes",
        "Titre RNCP", "Titres RNCP",
        "CQP", "CQPs",
        # ajouts
        "perfectionnement", "se perfectionner", "perfectionné",
        "spécialisation", "spécialiser", "spécialisé",
        "certifier", "certifiera",
        "actualiser", "actualisé",
        "recycler", "recyclage", "mettre à jour"
    ],

    "Nature": [
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

    "Durée": [
        "moins d’un an", "moins d'une année",
        "1-3 ans", "1 à 3 ans",
        "3-5 ans", "3 à 5 ans",
        "5-8 ans", "5 à 8 ans",
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
        "consigne", "consignes"
    ],

    "Diversité et quantité des postes à gérer": [
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
        "coaching", "coachings", "coach", "coachs",
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
        "briefing", "briefings",
        "reporting interne", "reportings internes",
        "animation réunion", "animations réunion",
        "synthèse", "synthèses",
        "négociation interne", "négociations internes",
        "facilitation", "facilitations",
        "communication transverse", "communications transverses",
        "gestion de conflit", "gestions de conflits",
        "culture feedback", "cultures feedback",
        # ajouts
        "informer", "information", "échanger", "échange",
        "communiquer", "communication", "dialoguer", "dialogue",
        "transmettre", "transmission", "expliquer", "explication",
        "coordonner", "coordination"
    ],

    "Nature des communications externes": [
        "relation client", "relations clients", "relations clientèle",
        "relations fournisseurs",
        "relations institutionnelles",
        "porte-parole", "porte-paroles",
        "représentation officielle", "représentations officielles",
        "communication de crise", "communications de crise",
        "relation presse", "relations presse",
        "lobbying",
        "community management", "community managements",
        "gestion partenariats", "gestions partenariats",
        # ajouts
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
        "langue des signes", "langues des signes",
        "bilingue", "bilingues",
        "trilingue", "trilingues",
        "multilingue", "multilingues",
        # ajouts
        "parler", "parlé", "lire", "lu", "rédiger", "rédigé",
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
        # ajouts
        "gérer", "gestion", "faire face", "affronter",
        "naviguer", "navigation", "complexifier", "complexification",
        "s'adapter", "adaptation", "volatil", "incertain", "ambigu", "turbulent"
    ],

    "Evolution de l'environnement": [
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
    if text2 is not None:
        text2 = text2.lower()
    if title in keywords:
        for keyword in keywords[title]:
            pattern = r'\b' + re.escape(keyword.lower()) + r's?\b'
            if re.search(pattern, text) and (re.search(pattern, text2) if text2 is not None else True):
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
    elif contains_keywords(row['sentence'],None, KEYWORDS, row['title']):
        sim += 0.5
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