import re
from typing import List, Dict

import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None


class Pretraitement:

    def __init__(self, cahier_text: str, questionnaire: pd.DataFrame | str,
                 spacy_model: str = "fr_dep_news_md") -> None:
        """
        :param cahier_text: Le texte brut du cahier des charges (full text)
        :param questionnaire: Un DataFrame du questionnaire
        :param spacy_model: Modèle spaCy à utiliser
        """
        self.cahier_text = cahier_text
        self.questionnaire = questionnaire
        self.spacy_model = spacy_model
        self._nlp = None

        # Outputs
        self.cahier_sentences_df: pd.DataFrame | None = None
        self.questionnaire_df: pd.DataFrame | None = None



    def _load_spacy(self) -> None:
        if self._nlp is not None:
            return

        if spacy is None:
            return  

        try:
            self._nlp = spacy.load(self.spacy_model, exclude=["ner", "tagger", "parser"])
        except OSError:

            self._nlp = spacy.blank("fr")
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")



    def load_questionnaire(self,
                           classe: str = "critereid",
                           sousClasse: str= "sscritereid",
                           question: str="questionnombre",
                           title_col: str = "titrequestion",
                           resp_col: str = "reponsedesc",
                           pts: str = "maximumpts") -> pd.DataFrame:
        """
        Prépare le questionnaire sous forme de DataFrame bien formaté
        """
        df = self.questionnaire
        expected = {classe,sousClasse,question, title_col, resp_col, pts}
        if not expected.issubset(df.columns):
            missing = expected - set(df.columns)
            raise ValueError(f"Colonnes manquantes dans le questionnaire : {missing}")

        df = df[[classe,sousClasse,question, title_col, resp_col, pts]].rename(columns={
            classe: "classe",
            sousClasse: "sousClasse",
            question: "question",
            title_col: "title",
            resp_col: "response",
            pts: "pts"
        })
        self.questionnaire_df = df
        return df

    def keep_essential_lines(self, lines: pd.DataFrame,selectedquestionsformodel) -> pd.DataFrame:
        """
        Garde uniquement les lignes necessaires
        """
        
        selectedquestionsformodel=selectedquestionsformodel.rename(columns={
            'critereid': 'classe',
            'sscritereid': 'sousClasse',
            'questionnombre':'question'})
        cols_communes=['classe','sousClasse','question']
        lines=lines.merge(selectedquestionsformodel,on=cols_communes,how='inner')
        return lines

    def sentencize_cahier(self, min_tokens: int = 2) -> List[str]:
        """
        Segmente le cahier de charges en phrases avec spaCy
        """
        self._load_spacy()
        doc = self._nlp(self.cahier_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        sentences = [s for s in sentences if len(s.split()) >= min_tokens]
        return sentences

    def sentencize_sentences(self, texts: List[str], min_tokens: int = 2) -> List[str]:
        """
        Découpage supplémentaire par ponctuation
        """
        sentences = []
        for text in texts:
            parts = [s.strip() for s in re.split(r'[.,:;/!?…\n\r\t]+', text) if s.strip()]
            sentences.extend(parts)
        sentences = [s for s in sentences if len(s.split()) >= min_tokens + 1]
        return sentences

    def build_cahier_df(self, **sent_kwargs) -> pd.DataFrame:
        """
        Construit le DataFrame des phrases du cahier
        """
        sents = self.sentencize_cahier(**sent_kwargs)
        df = pd.DataFrame({"sentence": sents})
        self.cahier_sentences_df = df
        return df
