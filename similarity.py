import numpy as np
import pandas as pd
import torch
from typing import List,Optional

from sentence_transformers import SentenceTransformer, InputExample



class Similarity:


    # --------------------------- INIT ---------------------------------
    def __init__(
        self,
        model_name: str = "dangvantuan/sentence-camembert-base",
        batch_size: int = 64,
    ) -> None:
        
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)



    

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Retourne des embeddings L2-normalisés."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )


    def _load_nlp(self) -> None:
        """Charge le modèle de lemmatisation."""
        if not hasattr(self, "_nlp"):
            try:
                import spacy

                self._nlp = spacy.load("fr_dep_news_trf", exclude=["ner", "tagger", "parser"])
                
            except ImportError:
                raise ImportError("Le module 'spacy' n'est pas installé.")
            except OSError:
                self._nlp = spacy.blank("fr")
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")


    def lemmatize(self, text: str) -> str:
        """Lemmatisation du texte."""
        if not hasattr(self, "_nlp"):
            self._load_nlp()
        doc = self._nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])
    

    # --------------------------- PUBLIC -------------------------------
    def top_k_matches(
        self,
        questions: List[str],
        corpus_sentences: List[str],
        *,
        k: int = 5,
        question_titles: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calcule la similarité cosinus et renvoie un DataFrame
        """

        q_vecs = self._encode(questions)
        
        s_vecs = self._encode(corpus_sentences)

        scores = np.matmul(q_vecs, s_vecs.T)  
        topk_idx = scores.argsort(axis=1)[:, -k:][:, ::-1] 

        rows: list[dict] = []
        for qi, q in enumerate(questions):
            for rank, sid in enumerate(topk_idx[qi], 1):
                # if float(scores[qi, sid])>=0.48:
                #     label="1"
                # elif float(scores[qi, sid])<=0.1:
                #     label="0"
                # else:
                #     label=""
                # if float(scores[qi, sid])>=0.78:
                #     label="1"

                # else:
                #     label="0"
                row = {
                    "question": q,
                    "sentence": corpus_sentences[sid],
                    "score": float(scores[qi, sid]),
                    "rank": rank,
                    # "label":label,
                }
                if question_titles is not None:
                    row["question_title"] = question_titles[qi]
                rows.append(row)

        df = pd.DataFrame(rows)
        if question_titles is not None:
            df = df[["question_title", "question", "sentence", "score", "rank"]]
        return df
