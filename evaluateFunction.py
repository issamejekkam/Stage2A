# Import necessary libraries
#--------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import sys
import spacy
import pymssql
import json
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import Classes
#--------------------------------------------------------------------------------------------------------------------------------------------------


from pretraitement import Pretraitement
from database import database
from similarity import Similarity



# load spacy model & database
#--------------------------------------------------------------------------------------------------------------------------------------------------


nlp = spacy.load("fr_core_news_md")  
# to change
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
conn=database("data.db")

conn.connect()
cursor=conn.cursor


# store parameters from command line arguments
#--------------------------------------------------------------------------------------------------------------------------------------------------


if len(sys.argv) > 1 :
    if len(sys.argv) > 1 and sys.argv[1]!="":
        CahierChargeNames = json.loads(sys.argv[1]) #list
        print(f"CahierChargeNames: {CahierChargeNames}")
    else:
        print("Error: CahierChargeName argument is missing.")
        sys.exit(1)
    if len(sys.argv) > 2 and sys.argv[2]!="":
        postesid = json.loads(sys.argv[2]) #list
    else:
        print("Error: posteid argument is missing.")
        sys.exit(1)

    if len(sys.argv) > 3 and sys.argv[3]!="":
        usersid = json.loads(sys.argv[3])
    else:
        print("Error: userid argument is missing.")
        sys.exit(1)

    if len(sys.argv) > 4 and sys.argv[4]!="":
        fonctionPostes = json.loads(sys.argv[4]) #list
    
    else:
        print("Error: fonctionPoste argument is missing.")
        sys.exit(1)


    if (len(sys.argv) > 5 and sys.argv[1]!=""):
        lexicales = json.loads(sys.argv[5]) #list
    else:
        print("Error: lexicaleorSemtique argument is missing.")
        sys.exit(1)
else:
    print("Usage: python evaluateFunction.py <CahierCharge>")
    sys.exit(1)



# Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------

# Function to lemmatize text using spaCy
def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Function to check if a text contains a verb
def contains_verb(text):
    doc = nlp(text)
    return any(token.pos_ == "VERB" or token.pos_ == "AUX" for token in doc)

# Function to check if a text contains at least two words
def twowordsmin(text):
    return len(text.split()) >= 2

# Function to control date format for SQL Server
def control_date(f_date: datetime) -> str:
    date_string = f_date.strftime("%d.%m.%Y")
    year = int(date_string[-4:])

    if year > 1890:
        return f"CONVERT(DATETIME, '{date_string}', 104)"
    else:
        return f"CONVERT(DATETIME, '{date_string}', 4)"


# Main processing loop
#--------------------------------------------------------------------------------------------------------------------------------------------------
for element in range(len(postesid)):
    # Extracting parameters for the current funtion
    posteid = postesid[element]
    fonctionPoste = fonctionPostes[element]
    fonctionid=posteid
    collid=posteid
    userid = usersid[element]
    CahierChargeName = CahierChargeNames[element]
    lexicale = lexicales[element]
    #printing parameters to check them
    print("-------------------------------------------------------------------")
    print(f"CahierChargeName: {CahierChargeName}")
    print(f"posteid: {posteid}")
    print(f"fonctionPoste: {fonctionPoste}")
    print(f"userid: {userid}")
    print(f"lexicale: {lexicale}")
    print("-------------------------------------------------------------------")




    
    if fonctionPoste=='POS':
        modele="mdlevaluation"
        table='postes'
        key='posteid'
        keyvalue=posteid
    elif fonctionPoste=='FCT':
        modele="mdlevaluation"
        table='fonctions'
        key='fctid'
        keyvalue=fonctionid
    if fonctionPoste=='POSREQ':
        modele="mdlcomp"
        table='postes'
        key='posteid'
        keyvalue=posteid
    elif fonctionPoste=='FCTREQ':
        modele="mdlcomp"
        table='fonctions'
        key='fctid'
        keyvalue=fonctionid
    elif fonctionPoste=='COL':
        modele="mdlcomp"
        table='collaborateurs'
        key='collid'
        keyvalue=posteid
    

    querytoselectmodele = f"SELECT {modele} FROM {table} WHERE {key} = %s"

    selectedmodele = pd.read_sql_query(querytoselectmodele, conn.connection, params=(keyvalue,))
    selectedmodele = selectedmodele.iloc[0, 0] if not selectedmodele.empty else None
    selectedquestionsformodelquery='''
        select * from questionsubmodels where mdlevaluation=%s   
    '''
    selectedquestionsformodel = pd.read_sql_query(selectedquestionsformodelquery, conn.connection, params=(selectedmodele,))
    selectedquestionsformodel=selectedquestionsformodel[['critereid','sscritereid','questionnombre']]
    for col in selectedquestionsformodel.columns:
        if col=='sscritereid' or col =='questionnombre' :
                selectedquestionsformodel[col] = pd.to_numeric(selectedquestionsformodel[col], errors='coerce').astype('Int64')



    # Pretraitement of documents
    #--------------------------------------------------------------------------------------------------------------------------------------------------

    # Get the current date to insert to lastupdated column
    now = datetime.now()
    sql_date_expr = control_date(now)

    # Read the "Questionnaire" from the database
    df = conn.readQuestionnaire()
    # Read the "Cahier de Charges" from the local repository named client/
    CahierCharge = conn.readCahierDeCharges(CahierChargeName)
    #Store the "questionnaire" into the variable questionnaire
    questionnaire = df
    # Initialize the Pretraitement class with the CahierCharge and questionnaire
    preproc = Pretraitement(CahierCharge, questionnaire)
    # Load the spaCy model if not already loaded (this is done in the beginning,just in case)
    preproc._load_spacy()
    # Load the questionnaire into a DataFrame
    df= preproc.load_questionnaire()
    
    # Keep only the lines that contain 'abcd' in the 'critereid' column
    df=preproc.keep_essential_lines(df,selectedquestionsformodel)
    # Preprocess the "Cahier de Charges" to get sentences using spaCy sentencization and storing them in a DataFrame 
    sentences = preproc.build_cahier_df()
    

    # Catch the five best matches for each long sentence (sentencization with spacy)
    #--------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize the Similarity class with a batch size of 32
    sim       = Similarity(batch_size=32)
  
    # This function finds the top-k most similar sentences from a corpus for each question using cosine similarity of their embeddings and returns the results as a pandas DataFrame
    matches   = sim.top_k_matches(
        questions=df["response"].tolist(),
        corpus_sentences=sentences["sentence"].tolist(),
        k=5,
        question_titles=df["title"].tolist(),
    )

    # Remove duplicates from the matches DataFrame based on question title and sentence
    matches_no_duplicates = matches.drop_duplicates(subset=["question_title", "sentence"])
    # Remove unnecessary columns from the matches DataFrame
    cols_to_drop = [col for col in ["score", "rank", "label"] if col in matches_no_duplicates.columns]
    # If there are any columns to drop, remove them from the DataFrame
    if cols_to_drop:
        matches_no_duplicates.drop(columns=cols_to_drop, inplace=True)

    # Create a mapping of question titles to sentences
    mapping={}
    # This code iterates over each row in matches_no_duplicates, grouping sentences by their question_title into the mapping dictionary
    for i, row in matches_no_duplicates.iterrows():
        question_title = row["question_title"]
        sentence = row["sentence"]
        if question_title not in mapping:
            mapping[question_title] = []
        mapping[question_title].append(sentence)

    #This code flattens a dictionary of titles to sentences into a DataFrame where each row contains a title and one of its sentences
    rows = []
    for title, sentences in mapping.items():
        for sentence in sentences:
            rows.append({'title': title, 'sentence': sentence})
    # resulting DataFrame is created from the rows list
    df_sentencized = pd.DataFrame(rows)


    # Catch the five best matches for each  sentence (sentencization with punctuation)
    #--------------------------------------------------------------------------------------------------------------------------------------------------


    # This code applies the sentencization using punctuation function to each list of sentences in the mapping dictionary
    mappingSentencized = {}
    for question_title, sentences_list in mapping.items():
        sentencized = preproc.sentencize_sentences(sentences_list)
        mappingSentencized[question_title] = sentencized
    # This code flattens the mappingSentencized dictionary into a DataFrame where each row contains a title and one of its sentences
    rows = []
    for title, sentences in mappingSentencized.items():
        for sentence in sentences:
            rows.append({'title': title, 'sentence': sentence})
    # resulting DataFrame is created from the rows list
    df_sentencized = pd.DataFrame(rows)

    # Remove sentences that do not meet the minimum word count requirement
    df_sentencized = df_sentencized[df_sentencized["sentence"].apply(twowordsmin)]


    df_without_verbs = df_sentencized[~df_sentencized["sentence"].apply(twowordsmin)]

    # Remove sentences that do not meet the minimum word count requirement from the mappingSentencized dictionary
    for title in mappingSentencized:
        mappingSentencized[title] = [s for s in mappingSentencized[title] if twowordsmin(s)]

    # Initialize the Similarity class again for the second round of matching
    sim       = Similarity(batch_size=32)   
    # Define the titles of the questions to be processed
    titles = df["title"].tolist()
    titles=list(set(titles))
    titles=sorted(titles)

    # Initialize an empty list to store all matches
    all_matches = []




    # Iterate over each title and find the top-k matches for the corresponding questions
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
    # Concatenate all matches into a single DataFrame
    matchesSentences = pd.concat(all_matches, ignore_index=True)



    # store the list of responses to all_matches and not_matched
    #--------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize an empty list to store sentences that have already been used
    sentences_used=[]
    # Initialize an empty list to store results for JSON output
    results_for_json = []
    # This code groups sentences by question title, selects top-scoring close matches per group, avoids duplicates, and appends relevant details to a results list for further processing
    for title, group in matchesSentences.groupby('question_title'):
    
        max_score = group['score'].max()
        # Do not neglict the sentences that have a score lower than 0.3 of the last choice
        threshold = 0.3
        close_matches = group[(group['score'] >= max_score - threshold) & (group['score'] <= max_score + threshold)].sort_values('score', ascending=False)
        close_matches = close_matches.drop_duplicates(subset=["sentence"])
        

        # Iterate over the close matches and append them to the results list if they haven't been used yet
        for _, row in close_matches.iterrows():
            if row['sentence'] not in sentences_used:
                sentences_used.append(row['sentence'])
            pts_value = df.loc[df["response"] == row["question"], "pts"]
            # Fill the results_for_json list with the relevant data
            results_for_json.append({
                "title": title,
                "question": row["question"],
                "sentence": row["sentence"],
                "score": row["score"],
                "pts": pts_value.values[0] 
            })

    # This variable will be user in the sematic if block
    results_for_json_to_use_for_semantique=results_for_json
    if title == titles[-1]: 
        # If the title is the last one in the list, we proceed to store the results in the database
        jsoncontentdf= pd.DataFrame(results_for_json)

        for row in jsoncontentdf.itertuples(index=False):
            # Check if the row already exists in the all_matches table and update it if it does, otherwise insert a new row
            cursor.execute("""
                IF EXISTS (
                    SELECT 1 FROM all_matches
                    WHERE filename=%s AND title=%s AND question=%s AND sentence=%s
                )
                BEGIN
                    UPDATE all_matches
                    SET score = %s, pts = %s
                    WHERE filename=%s AND title=%s AND question=%s AND sentence=%s
                END
                ELSE
                BEGIN
                    INSERT INTO all_matches (filename, title, question, sentence, score, pts)
                    VALUES (%s, %s, %s, %s, %s, %s)
                END
            """, (
                f"{CahierChargeName}", row.title, row.question, row.sentence,  # for EXISTS and UPDATE WHERE
                row.score, row.pts,                                            # UPDATE SET
                f"{CahierChargeName}", row.title, row.question, row.sentence,  # again for UPDATE WHERE
                f"{CahierChargeName}", row.title, row.question, row.sentence,  # for INSERT
                row.score, row.pts                                              # for INSERT
            ))




        # Commit the changes to the database
        conn.commit()

    # Store the unmatched sentences in the not_matched table
    sentences = []
    for i in mappingSentencized:
        for j in mappingSentencized[i]:
            if j not in sentences:
                sentences.append(j)
    # The values that are not in the sentences_used list will be stored in the not_matched table
    to_use = [s for s in sentences if s not in sentences_used]

  

    for row in to_use:
        # Check if the row already exists in the not_matched table and insert it if it does not
        cursor.execute('''
            SELECT COUNT(*) FROM not_matched WHERE filename = %s AND sentence = %s
        ''', (CahierChargeName, row))
        
        exists = cursor.fetchone()[0]

        if not exists:
            cursor.execute('''
                INSERT INTO not_matched (filename, sentence) VALUES (%s, %s)
            ''', (CahierChargeName, row))
    # Commit the changes to the database
    conn.commit()
    # Convert the results_for_json list to a DataFrame for further processing
    results_for_json = pd.DataFrame(results_for_json)



    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    # -------------------------semantique----------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------



    
    if lexicale == "false":
        # Sort the results_for_json DataFrame by score in descending order
        results_for_json = results_for_json.sort_values(by='score', ascending=False)
        # Remove duplicate titles, keeping the first occurrence
        # That will take the best match for each title semantically based on the score of semantic similarity
        results_for_json = results_for_json.drop_duplicates(subset=['title'], keep='first')  
        # Add 'codeReponse' and 'codeQuestion' columns to the results_for_json DataFrame
        if 'codeReponse' not in results_for_json.columns:
            results_for_json.insert(3, 'codeReponse', '') 
        if 'codeQuestion' not in results_for_json.columns:
            results_for_json.insert(1, 'codeQuestion', '')
        # Read the cleaned questionnaire to merge it the questionnaire on the reponsedesc column
        questionnaireClean = conn.readQuestionnaire()

        for i in results_for_json.index:
            for index, row in questionnaireClean.iterrows():
    
                # Check if the reponsedesc in the questionnaire matches the question in results_for_json
                if row["reponsedesc"] == results_for_json.at[i, 'question']:
                    # If it matches, create the codeQuestion and codeReponse
                        a = row['critereid']
                        b = row['sscritereid']
                        c = row['questionnombre']
                        d = row['reponse']
                        results_for_json.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                        results_for_json.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"
        # Rename the columns for better readability
        results_for_json=results_for_json.rename(columns={
            'title': 'Question',
            'question': 'niv de réponse'})
        

        # Inference
        #--------------------------------------------------------------------------------------------------------------------------------------------------



        # Get the results of the semantic similarity that have a score lower than 0.6
        resultalLowScoreAndSimilarity = results_for_json[results_for_json['score'] < 0.6]
        # list the codeReponse of the results that have a low score
        codelow = list(resultalLowScoreAndSimilarity['codeReponse'].sort_values())
 
        # Split each element of resultalLowScoreAndSimilarityCodeQuestion by '-' and '.' and store as sublists
        codelowlisted = [re.split(r'[-.]', item) for item in codelow]

        # Get all the codeReponse from the results_for_json DataFrame
        code = list(results_for_json['codeReponse'].sort_values())
        # Split each element of code by '-' and '.' and store as sublists
        codelisted = [re.split(r'[-.]', item) for item in code]

        # Filter out the elements in codelisted that are not in codelowlisted
        codehighlisted = [item for item in codelisted if item not in codelowlisted]
        # Get only the codeQuestion part of the codeReponse of the codelowlisted elements
        codelowlisted=[item[:3] for item in codelowlisted]



        # The goal here is to first retrieve the fctid from the database that matches the same formation level,
        # then iterate through codehighlisted to find the fctid that best matches my function.
        #---------------------------------------------------------------------------------------------------------------------------------------------------


        # Query to get the fctid from the database based on the critereid, sscritereid, questionnombre, and reponsenivmin that match the formation de base of my function
        if fonctionPoste=='POS':
            fonctionOuPoste="posteid"
            tableinference='questionreponseposte'

        elif fonctionPoste=='FCT':
            fonctionOuPoste="fctid"
            tableinference='questionreponsefonction'

        if fonctionPoste=='POSREQ':
            fonctionOuPoste="posteid"
            tableinference='evaluationpostecomp'

        elif fonctionPoste=='FCTREQ':
            fonctionOuPoste="fctid"
            tableinference='evaluationfctcomp'

        elif fonctionPoste=='COL':
            fonctionOuPoste="collid"
            tableinference='collresponsequestion'

        
        stats_template = f"""
        SELECT {fonctionOuPoste} 
        FROM {tableinference} 
        WHERE critereid = %s AND sscritereid = %s AND questionnombre = %s AND reponsenivmin = %s
        """
        # Initialize an empty list to store the filtered fctid
        filtered_fctid = []

        if codehighlisted:
            # Step 1: Get the initial fctid list based on the first element of codehighlisted
            critereid, sscritereid, questionnombre, reponsenivmin = codehighlisted[0]
            values = (critereid, sscritereid, questionnombre, reponsenivmin)
            initial_df = pd.read_sql_query(stats_template, conn.connection, params=values)
            
            LIST = initial_df[f'{fonctionOuPoste}'].tolist()

            if LIST:
            # Step 2: Iterate through the remaining elements of codehighlisted
                for i in range(1, len(codehighlisted)):
                    if not LIST:
                        continue  # continue  if there's no match left
                    
                    critereid, sscritereid, questionnombre, reponsenivmin = codehighlisted[i]
                    # fctidt that matched the last iteration
                    placeholders = ','.join(['%s'] * len(LIST))
                    stats = f"""
                        SELECT {fonctionOuPoste} 
                        FROM {tableinference}  
                        WHERE critereid = %s AND sscritereid = %s AND questionnombre = %s 
                        AND reponsenivmin = %s AND {fonctionOuPoste} IN ({placeholders})
                    """
                    values = (critereid, sscritereid, questionnombre, reponsenivmin, *LIST)
                    
                    filtered_df = pd.read_sql_query(stats, conn.connection, params=values)
                    if filtered_df.empty:
                        continue
                    else:
                        LIST = filtered_df[f'{fonctionOuPoste}'].tolist()


            # Final Result
            filtered_fctid = LIST
        if filtered_fctid:
            # Get the evaluation of the final fctid
            stats_template = f"""
            SELECT * 
            FROM {tableinference}  
            WHERE {fonctionOuPoste}= %s
            """
            fctid = pd.Series(filtered_fctid).unique().tolist()[0]
            values = (fctid)
            # Execute the SQL query to get the evaluation of the final fctid
            theOne = pd.read_sql_query(stats_template, conn.connection, params=values)
            # Add 'codeReponse' and 'codeQuestion' columns to theOne DataFrame
            if 'codeReponse' not in theOne.columns:
                theOne.insert(3, 'codeReponse', '') 
            if 'codeQuestion' not in theOne.columns:
                theOne.insert(1, 'codeQuestion', '')

            # Create the codeQuestion and codeReponse for each row in theOne DataFrame
            for i in theOne.index:
                a = theOne.loc[i, 'critereid']
                b = theOne.loc[i, 'sscritereid']
                c = theOne.loc[i, 'questionnombre']
                d = theOne.loc[i, 'Reponsenivmin']
                # Convert float to int, then to string (removes decimal part)
                d = str(int(d))
                theOne.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                theOne.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"

            # Get the codeQuestion from the codeReponse DataFrame
            codelowQuestion=[item[:5] for item in codelow]
            finalResponses=[]
            for i in theOne.index:
                if theOne.at[i, 'codeQuestion'] in codelowQuestion:
                    finalResponses.append(theOne.at[i, 'codeReponse'])
            # Remove duplicates from finalResponses    
            finalResponses = set(finalResponses)
            # Split each element of finalResponses by '-' and '.' and store as sublists
            coderepfinalesplitted = [re.split(r'[-.]', item) for item in finalResponses]
            # Create a DataFrame from the coderepfinalesplitted list
            dfToReplace = pd.DataFrame(
                coderepfinalesplitted,
                columns=['critereid', 'sscritereid', 'questionnombre', 'reponsenivmin']
            )

            dfToReplace['reponsenivmax'] = dfToReplace['reponsenivmin']
            
            dfToReplace = dfToReplace.drop_duplicates(subset=['critereid', 'sscritereid', 'questionnombre'])


        # Get the codeReponse from the results_for_json DataFrame
        resultats_finale=results_for_json['codeReponse']
        # Sort the resultats_finale DataFrame by codeReponse
        resultats_finale_sorted= resultats_finale.sort_values(ascending=True).reset_index(drop=True)

        # Step 1: split the resultats_finale_sorted by '-' to get the critereid, sscritereid, questionnombre, reponsenivmin
        temp = resultats_finale_sorted.str.split('-', expand=True)

        # Step 2: split the second part of temp by '.' to get the sscritereid, questionnombre, reponsenivmin
        left_parts = temp[1].str.split('.', expand=True)

        # Create the toStored DataFrame based on the fonctionPoste
        if fonctionPoste=='POS' or fonctionPoste=='POSREQ':
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
        elif fonctionPoste=="FCT" or fonctionPoste=='FCTREQ':
            toStored = pd.DataFrame({
            'fctid': fonctionid,
            'evalid': 200,
            'critereid': temp[0],
            'sscritereid': left_parts[0],
            'questionnombre': left_parts[1],
            'Reponsenivmin': left_parts[2],
            'Reponsenivmax': left_parts[2],
            'desactive': 0,
            'lastupdated': None,
            'usrid': userid})
        elif fonctionPoste=='COL':
            toStored = pd.DataFrame({
            'collid': collid,
            'evalid': 200,
            'critereid': temp[0],
            'sscritereid': left_parts[0],
            'questionnombre': left_parts[1],
            'Reponsenivmin': left_parts[2],
            'Reponsenivmax': left_parts[2],
            'desactive': 0,
            'lastupdated': None,
            'usrid': userid})

        # Col to be added
        if filtered_fctid:
        # Rename the columns to match the database schema
            dfToReplace.rename(columns={
            'reponsenivmin': 'Reponsenivmin',  
            'reponsenivmax': 'Reponsenivmax'})

            # Merge the toStored DataFrame with dfToReplace DataFrame on critereid, sscritereid, questionnombre
            # This will update the toStored DataFrame with the values from dfToReplace
            key_columns = ['critereid', 'sscritereid', 'questionnombre']
            toStored.set_index(key_columns, inplace=True)   
            dfToReplace.set_index(['critereid', 'sscritereid', 'questionnombre'], inplace=True)
            toStored.update(dfToReplace)
            toStored.reset_index(inplace=True)

        # Export toStored DataFrame to JSON
        tostored_json = toStored.to_json(orient="records", force_ascii=False, indent=4)
        with open(f"results/semantique_of__{CahierChargeName}.json", "w", encoding="utf-8") as f:

            f.write(tostored_json)
            print("JSON file created successfully.")


        # Insert or update the toStored DataFrame in the database
        if fonctionPoste is not None:
            criteres=[]
            # Responsabilités poste
            if fonctionPoste == "POS":
                
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first 
                    cursor.execute('''

                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationposte
                            WHERE posteid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationposte (
                                posteid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.posteid, row.evalid, row.posteid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM questionreponseposte
                            WHERE posteid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO questionreponseposte (
                            posteid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the posteid and evalid for further processing
                    posteid = row.posteid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationposte table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN questionreponseposte e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.posteid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(posteid, evalid))
                criteres=list(set(criteres))


                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                # Get the niveauid based on the total points
                # This query retrieves the niveauid from the niveauxeval table where the total points fall within
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationposte table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationposte WHERE posteid = %s AND evalid = %s
                )
                    UPDATE evaluationposte
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE posteid = %s AND evalid = %s
                
                ''', (
                    posteid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, posteid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

            elif fonctionPoste == "POSREQ":
                # Compétences poste
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationscompposte
                            WHERE posteid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationscompposte (
                                posteid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.posteid, row.evalid, row.posteid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationpostecomp
                            WHERE posteid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s 
                        )
                        INSERT INTO evaluationpostecomp (
                            posteid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the posteid and evalid for further processing
                    posteid = row.posteid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompposte table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN evaluationpostecomp e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.posteid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(posteid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                # Get the niveauid based on the total points
                # This query retrieves the niveauid from the niveauxeval table where the total points fall within
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompposte table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationscompposte WHERE posteid = %s AND evalid = %s
                )
                    UPDATE evaluationscompposte
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE posteid = %s AND evalid = %s
                
                ''', (
                    posteid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, posteid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

                    
            elif fonctionPoste == "FCT":
                # Responsabilités fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationfct
                            WHERE fctid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationfct (
                                fctid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.fctid, row.evalid, row.fctid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM questionreponsefonction
                            WHERE fctid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        BEGIN
                            INSERT INTO questionreponsefonction (
                                fctid, evalid, critereid, sscritereid, questionnombre,
                                Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        END
                    ''', (
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre, # pour WHERE
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,  # pour INSERT
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    fonctionid = row.fctid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationfct table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN questionreponsefonction e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.fctid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(fonctionid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationfct WHERE fctid = %s AND evalid = %s
                )
                    UPDATE evaluationfct
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE fctid = %s AND evalid = %s
                
                ''', (
                    fonctionid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, fonctionid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

                    
            #here
            elif fonctionPoste == "FCTREQ":
                # Compétences fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationscompfct
                            WHERE fctid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationscompfct (
                                fctid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.fctid, row.evalid, row.fctid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table               
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationfctcomp
                            WHERE fctid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO evaluationfctcomp (
                            fctid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))  
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    fonctionid = row.fctid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompfct table
                # This query retrieves the total points for each critereid based on the reponsebase    
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN evaluationfctcomp e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.fctid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(fonctionid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationscompfct WHERE fctid = %s AND evalid = %s
                )
                    UPDATE evaluationscompfct
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE fctid = %s AND evalid = %s
                
                ''', (
                    fonctionid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, fonctionid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()
            elif fonctionPoste == "COL":
                # Compétences fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM collevaluation
                            WHERE collid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO collevaluation (
                                collid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.collid, row.evalid, row.collid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table               
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM collresponsequestion
                            WHERE collid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO collresponsequestion (
                            collid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.collid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.collid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))  
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    collid = row.collid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompfct table
                # This query retrieves the total points for each critereid based on the reponsebase    
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN collresponsequestion e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.collid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(collid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM collevaluation WHERE collid = %s AND evalid = %s
                )
                    UPDATE collevaluation
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE collid = %s AND evalid = %s
                
                ''', (
                    collid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, collid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()



            conn.commit()


    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    # -------------------------lexicale----------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------



    if lexicale=="true":



        # Read the keywords from the database
        KEYWORDS = conn.readkeywords()
        # transform the all_matches results to a DataFrame
        resultats=pd.DataFrame(results_for_json_to_use_for_semantique)

        # Check if contains any keywords
        def contains_keywords(text, text2, keywords, title):
            text = text.lower()
            if text2 is not None:
                text2 = text2.lower()
            if title in keywords:
                for keyword in keywords[title]:
                    pattern = r'\b' + re.escape(keyword.lower()) + r's?\b'
                    if re.search(pattern, text) and (re.search(pattern, text2) if text2 is not None else True):
                        return True
            return False

        # lower case 
        def to_lower(text):
            return text.lower()

        # cosine similarity using TF-IDF
        def tfidf_similarity(text1, text2):
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([text1, text2])

            return cosine_similarity(tfidf[0], tfidf[1])[0][0]

        # lemmatization
        resultats["sentence_lemmatized"] = resultats["sentence"].apply(to_lower)
        resultats["sentence_lemmatized"] = resultats["sentence_lemmatized"].apply(lemmatize)
        resultats['question_lemmatized'] = resultats['question'].apply(to_lower)
        resultats['question_lemmatized'] = resultats['question_lemmatized'].apply(lemmatize)

        # Calculate the similarity between the sentence and the question lemmatized
        similarities = []
        for idx, row in resultats.iterrows():
            sim = tfidf_similarity(row['sentence_lemmatized'], row['question_lemmatized'])
            # Check if the sentence and question contains keywords
            if contains_keywords(row['sentence'], row['question'], KEYWORDS, row['title']):
                # If both contain keywords, add 1 to the similarity score
                sim += 1
            # If only the sentence contains keywords, add 0.5 to the similarity score
            elif contains_keywords(row['sentence'],None, KEYWORDS, row['title']):
                sim += 0.5
            
            similarities.append(sim)
        # Add the similarity scores to the resultats DataFrame
        resultats['similarity'] = similarities
        # Sort the resultats DataFrame by similarity in descending order
        resultats=resultats.sort_values(by='similarity', ascending=False)
        # Select the necessary columns for the final result
        resultats_needed = resultats[["title", "question", "sentence", "score", "pts", "similarity"]]
        # Sort the resultats_needed DataFrame by similarity in descending order
        resultats_needed.sort_values(by='similarity', ascending=False, inplace=True)
        # Remove duplicates based on the 'title' column, keeping the first occurrence
        resultats_fin = resultats_needed.sort_values('similarity', ascending=False).drop_duplicates(subset=['title'], keep='first')
        # Read the questionnaireClean DataFrame from the database
        questionnaireClean = conn.readQuestionnaire()
        # Add 'codeReponse' and 'codeQuestion' columns to the resultats_fin DataFrame
        for i in resultats_fin.index:
            for index, row in questionnaireClean.iterrows():
                if 'codeReponse' not in resultats_fin.columns:
                    resultats_fin.insert(3, 'codeReponse', '') 
                if 'codeQuestion' not in resultats_fin.columns:
                    resultats_fin.insert(1, 'codeQuestion', '') 
                
                # Create the codeQuestion and codeReponse for each row in resultats_fin DataFrame
                if row["reponsedesc"] == resultats_fin.at[i, 'question']:
                        a = row['critereid']
                        b = row['sscritereid']
                        c = row['questionnombre']
                        d = row['reponse']
                        resultats_fin.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                        resultats_fin.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"
        # Rename the columns of resultats_fin DataFrame
        resultats_fin=resultats_fin.rename(columns={
            'title': 'Question',
            'question': 'niv de réponse'})

        # Inference
        # ------------------------------------------------------------------

        # Filter the resultats_fin DataFrame to get the codes with low score and similarity
        resultalLowScore = resultats_fin[resultats_fin['score'] < 0.6]

        resultalLowScoreAndSimilarity = resultalLowScore[resultalLowScore['similarity'] < 0.5]
        # Sort the resultalLowScoreAndSimilarity DataFrame by codeReponse
        codelow = list(resultalLowScoreAndSimilarity['codeReponse'].sort_values())
        # Split each element of resultalLowScoreAndSimilarityCodeQuestion by '-' and '.' and store as sublists
        codelowlisted = [re.split(r'[-.]', item) for item in codelow]
        # Get all the codes from resultats_fin DataFrame
        code = list(resultats_fin['codeReponse'].sort_values())
        # Split each element of code by '-' and '.' and store as sublists
        codelisted = [re.split(r'[-.]', item) for item in code]
        # Filter the codes that are not in codelowlisted
        codehighlisted = [item for item in codelisted if item not in codelowlisted]
        # Get the codeQuestion from the codeReponse DataFrame
        codelowlisted=[item[:3] for item in codelowlisted]


        #----------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # Same logic as semantic
        # For all the rest of the code 
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------- 
        if fonctionPoste=='POS':
            fonctionOuPoste="posteid"
            tableinference='questionreponseposte'

        elif fonctionPoste=='FCT':
            fonctionOuPoste="fctid"
            tableinference='questionreponsefonction'

        if fonctionPoste=='POSREQ':
            fonctionOuPoste="posteid"
            tableinference='evaluationpostecomp'

        elif fonctionPoste=='FCTREQ':
            fonctionOuPoste="fctid"
            tableinference='evaluationfctcomp'

        elif fonctionPoste=='COL':
            fonctionOuPoste="collid"
            tableinference='collresponsequestion'

        
        stats_template = f"""
        SELECT {fonctionOuPoste} 
        FROM {tableinference} 
        WHERE critereid = %s AND sscritereid = %s AND questionnombre = %s AND reponsenivmin = %s
        """
        filtered_fctid = []

        if codehighlisted:
            critereid, sscritereid, questionnombre, reponsenivmin = codehighlisted[0]
            values = (critereid, sscritereid, questionnombre, reponsenivmin)
            initial_df = pd.read_sql_query(stats_template, conn.connection, params=values)
            LIST = initial_df[f'{fonctionOuPoste}'].tolist()

          
            if LIST:
                for i in range(1, len(codehighlisted)):
                    if not LIST:
                        continue  
                    
                    critereid, sscritereid, questionnombre, reponsenivmin = codehighlisted[i]

                    placeholders = ','.join(['%s'] * len(LIST))
                    stats = f"""
                        SELECT {fonctionOuPoste} 
                        FROM {tableinference}  
                        WHERE critereid = %s AND sscritereid = %s AND questionnombre = %s 
                        AND reponsenivmin = %s AND {fonctionOuPoste} IN ({placeholders})
                    """
                    values = (critereid, sscritereid, questionnombre, reponsenivmin, *LIST)
                    filtered_df = pd.read_sql_query(stats, conn.connection, params=values)
                    if filtered_df.empty:
                        continue
                    else:
                        LIST = filtered_df[f'{fonctionOuPoste}'].tolist()


            # Résultat final
            filtered_fctid = LIST
        if filtered_fctid:
            stats_template = f"""
            SELECT * 
            FROM {tableinference}  
            WHERE {fonctionOuPoste}= %s
            """
            fctid = pd.Series(filtered_fctid).unique().tolist()[0]
            values = (fctid)
            theOne = pd.read_sql_query(stats_template, conn.connection, params=values)

            if 'codeReponse' not in theOne.columns:
                theOne.insert(3, 'codeReponse', '') 
            if 'codeQuestion' not in theOne.columns:
                theOne.insert(1, 'codeQuestion', '')


            for i in theOne.index:
                a = theOne.loc[i, 'critereid']
                b = theOne.loc[i, 'sscritereid']
                c = theOne.loc[i, 'questionnombre']
                d = theOne.loc[i, 'Reponsenivmin']
                # Convert float to int, then to string (removes decimal part)
                d = str(int(d))
                theOne.at[i, 'codeQuestion'] = f"{a}-{b}.{c}"
                theOne.at[i, 'codeReponse'] = f"{a}-{b}.{c}.{d}"

            codelowQuestion=[item[:5] for item in codelow]

            finalResponses=[]
            for i in theOne.index:
                if theOne.at[i, 'codeQuestion'] in codelowQuestion:
                    finalResponses.append(theOne.at[i, 'codeReponse'])
                    
            finalResponses = set(finalResponses)
            coderepfinalesplitted = [re.split(r'[-.]', item) for item in finalResponses]
            dfToReplace = pd.DataFrame(
                coderepfinalesplitted,
                columns=['critereid', 'sscritereid', 'questionnombre', 'reponsenivmin']
            )
            dfToReplace['reponsenivmax'] = dfToReplace['reponsenivmin']
            dfToReplace = dfToReplace.drop_duplicates(subset=['critereid', 'sscritereid', 'questionnombre'])
    
        conn.commit()
        resultats_finale=resultats_fin['codeReponse']

        resultats_finale_sorted= resultats_finale.sort_values(ascending=True).reset_index(drop=True)
        resultats_finale_sorted


        temp = resultats_finale_sorted.str.split('-', expand=True)

        left_parts = temp[1].str.split('.', expand=True)

        if fonctionPoste=='POS' or fonctionPoste=='POSREQ':
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
        elif fonctionPoste=="FCT" or fonctionPoste=='FCTREQ':
            toStored = pd.DataFrame({
            'fctid': fonctionid,
            'evalid': 200,
            'critereid': temp[0],
            'sscritereid': left_parts[0],
            'questionnombre': left_parts[1],
            'Reponsenivmin': left_parts[2],
            'Reponsenivmax': left_parts[2],
            'desactive': 0,
            'lastupdated': None,
            'usrid': userid})
        elif fonctionPoste=="COL":
            toStored = pd.DataFrame({
            'collid': collid,
            'evalid': 200,
            'critereid': temp[0],
            'sscritereid': left_parts[0],
            'questionnombre': left_parts[1],
            'Reponsenivmin': left_parts[2],
            'Reponsenivmax': left_parts[2],
            'desactive': 0,
            'lastupdated': None,
            'usrid': userid})


        if filtered_fctid:
            dfToReplace.rename(columns={
            'reponsenivmin': 'Reponsenivmin',  
            'reponsenivmax': 'Reponsenivmax'},inplace=True)

            key_columns = ['critereid', 'sscritereid', 'questionnombre']
            toStored.set_index(key_columns, inplace=True)   
            dfToReplace.set_index(['critereid', 'sscritereid', 'questionnombre'], inplace=True)
            toStored.update(dfToReplace)
            toStored.reset_index(inplace=True)


        tostored_json = toStored.to_json(orient="records", force_ascii=False, indent=4)
        with open(f"results/lexicale_of_{CahierChargeName}.json", "w", encoding="utf-8") as f:
            f.write(tostored_json)
            print("JSON file created successfully.")
        # Insert or update the toStored DataFrame in the database
                # Insert or update the toStored DataFrame in the database
        if fonctionPoste is not None:
            criteres=[]
            # Responsabilités poste
            if fonctionPoste == "POS":
                
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first 
                    cursor.execute('''

                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationposte
                            WHERE posteid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationposte (
                                posteid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.posteid, row.evalid, row.posteid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM questionreponseposte
                            WHERE posteid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO questionreponseposte (
                            posteid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the posteid and evalid for further processing
                    posteid = row.posteid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationposte table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN questionreponseposte e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.posteid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(posteid, evalid))
                criteres=list(set(criteres))


                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                # Get the niveauid based on the total points
                # This query retrieves the niveauid from the niveauxeval table where the total points fall within
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationposte table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationposte WHERE posteid = %s AND evalid = %s
                )
                    UPDATE evaluationposte
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE posteid = %s AND evalid = %s
                
                ''', (
                    posteid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, posteid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

            elif fonctionPoste == "POSREQ":
                # Compétences poste
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationscompposte
                            WHERE posteid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationscompposte (
                                posteid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.posteid, row.evalid, row.posteid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationpostecomp
                            WHERE posteid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s 
                        )
                        INSERT INTO evaluationpostecomp (
                            posteid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.posteid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the posteid and evalid for further processing
                    posteid = row.posteid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompposte table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN evaluationpostecomp e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.posteid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(posteid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                # Get the niveauid based on the total points
                # This query retrieves the niveauid from the niveauxeval table where the total points fall within
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompposte table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationscompposte WHERE posteid = %s AND evalid = %s
                )
                    UPDATE evaluationscompposte
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE posteid = %s AND evalid = %s
                
                ''', (
                    posteid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, posteid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

                    
            elif fonctionPoste == "FCT":
                # Responsabilités fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationfct
                            WHERE fctid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationfct (
                                fctid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.fctid, row.evalid, row.fctid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM questionreponsefonction
                            WHERE fctid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        BEGIN
                            INSERT INTO questionreponsefonction (
                                fctid, evalid, critereid, sscritereid, questionnombre,
                                Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        END
                    ''', (
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre, # pour WHERE
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,  # pour INSERT
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    fonctionid = row.fctid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationfct table
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN questionreponsefonction e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.fctid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(fonctionid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationfct WHERE fctid = %s AND evalid = %s
                )
                    UPDATE evaluationfct
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE fctid = %s AND evalid = %s
                
                ''', (
                    fonctionid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, fonctionid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()

                    
            #here
            elif fonctionPoste == "FCTREQ":
                # Compétences fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationscompfct
                            WHERE fctid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO evaluationscompfct (
                                fctid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.fctid, row.evalid, row.fctid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table               
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM evaluationfctcomp
                            WHERE fctid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO evaluationfctcomp (
                            fctid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.fctid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))  
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    fonctionid = row.fctid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompfct table
                # This query retrieves the total points for each critereid based on the reponsebase    
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN evaluationfctcomp e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.fctid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(fonctionid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM evaluationscompfct WHERE fctid = %s AND evalid = %s
                )
                    UPDATE evaluationscompfct
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE fctid = %s AND evalid = %s
                
                ''', (
                    fonctionid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, fonctionid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()
            elif fonctionPoste == "COL":
                # Compétences fonction
                for row in toStored.itertuples(index=False):
                    # Insert the "consolidé" table first
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM collevaluation
                            WHERE collid = %s AND evalid = %s 
                        )
                        BEGIN
                            INSERT INTO collevaluation (
                                collid, evalid, usrid,lastupdated
                            ) VALUES (%s, %s,%s,%s)
                        END
                                    ''',(
                                        row.collid, row.evalid, row.collid, row.evalid,row.usrid,now
                                    ))
                    # Insert to the detailed table               
                    cursor.execute('''
                        IF NOT EXISTS (
                            SELECT 1 FROM collresponsequestion
                            WHERE collid = %s AND evalid = %s AND critereid = %s AND sscritereid = %s AND questionnombre = %s
                        )
                        INSERT INTO collresponsequestion (
                            collid, evalid, critereid, sscritereid, questionnombre,
                            Reponsenivmin, Reponsenivmax, desactive, lastupdated, usrid
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        row.collid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.collid, row.evalid, row.critereid, row.sscritereid, row.questionnombre,
                        row.Reponsenivmin, row.Reponsenivmax, row.desactive, now, row.usrid
                    ))  
                    criteres.append(row.critereid)

                    # Commit the changes to the database
                    conn.commit()
                    # Store the fonctionid and evalid for further processing
                    collid = row.collid
                    evalid = row.evalid
                # Calculate the total points for each critereid and update the evaluationscompfct table
                # This query retrieves the total points for each critereid based on the reponsebase    
                df_points = pd.read_sql_query("""
                    SELECT e.critereid, SUM(r.maximumpts) AS total_point
                    FROM reponsebase r
                    JOIN collresponsequestion e
                        ON r.critereid = e.critereid
                        AND r.sscritereid = e.sscritereid
                        AND r.questionnombre = e.questionnombre
                        AND r.reponse = e.reponsenivmin
                    WHERE
                        e.collid = %s
                        AND e.evalid = %s
                    GROUP BY
                        e.critereid
                """, conn.connection, params=(collid, evalid))
                criteres=list(set(criteres))

                critere_scores = {}

                for critere in criteres:
                    row = df_points.loc[df_points['critereid'] == critere, 'total_point']
                    if not row.empty:
                        critere_scores[critere] = int(row.values[0])

                critere_scores=dict(sorted(critere_scores.items()))
                valueToadd = ''.join(f"{key}[{val}]" for key, val in critere_scores.items())
                totalPoints = sum(critere_scores.values())
                df_niv = pd.read_sql_query("""
                select niveauid from niveauxeval where %s>=Minpts and Maxpts>=%s
                """, conn.connection, params=(totalPoints, totalPoints))
                niveauid = df_niv['niveauid'].values[0] if not df_niv.empty else None
                # Update the evaluationscompfct table with the total points and niveauid
                cursor.execute('''
                IF EXISTS (
                    SELECT 1 FROM collevaluation WHERE collid = %s AND evalid = %s
                )
                    UPDATE collevaluation
                    SET TtPtsMin = %s, TtPtsMax = %s, Nivmin = %s, Nivmax = %s,
                        totalcrit_min = %s, totalcrit_Max = %s
                    WHERE collid = %s AND evalid = %s
                
                ''', (
                    collid, evalid,  # For IF EXISTS
                    totalPoints, totalPoints, niveauid, niveauid, valueToadd, valueToadd, collid, evalid,  # For UPDATE
                ))
                # Commit the changes to the database
                conn.commit()



            



            conn.commit()


    conn.commit()

    print(f"Evaluation function {element} completed successfully.")

conn.commit()
conn.close()