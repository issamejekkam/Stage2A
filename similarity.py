from __future__ import annotations 
from typing import List, Optional
import numpy as np
import pandas as pd
import torch

from needed import return_train_loader
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses,InputExample
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss


import json
from torch.utils.data import DataLoader


class Similarity:


    # --------------------------- INIT ---------------------------------
    def __init__(
        self,
        model_name: str = "dangvantuan/sentence-camembert-base",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)


    # def train(self, triplet_json="Triplets.json", epochs=10, out_dir="models/camembert_mnr_v1"):
    #     data = json.load(open(triplet_json, encoding="utf-8"))

    #     pairs = []
    #     for item in data:
    #         anchor = item["question"]
    #         for pos in item["positive"]:
    #             pairs.append(InputExample(texts=[anchor, pos]))

    #     train_loader = DataLoader(pairs, shuffle=True, batch_size=self.batch_size)
    #     train_loss   = MultipleNegativesRankingLoss(self.model)

    #     self.model.fit(
    #         train_objectives=[(train_loader, train_loss)],
    #         epochs=epochs,
    #         show_progress_bar=True,
    #         output_path=out_dir
    #     )
    #     # recharge le modèle fine-tuné
    #     self.model = SentenceTransformer(out_dir, device=self.device)

    

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
        k: int = 2,
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
