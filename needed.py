from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import pandas as pd





def return_train_loader(triplets: list[dict]) -> DataLoader:
    """
    Retourne le DataLoader pour l'entraînement.
    """
    rows = []                          
    train_examples = []                

    for t in triplets:
        anchor = t["question"]

        
        positives = t["positive"] if isinstance(t["positive"], list) else [t["positive"]]
        negatives = t["negatives"] if isinstance(t["negatives"], list) else [t["negatives"]]

        for pos in positives:
            for neg in negatives:
                train_examples.append(InputExample(texts=[anchor, pos, neg]))

                rows.append({"anchor": anchor, "positive": pos, "negative": neg})

    # DataFrame propre
    train_df = pd.DataFrame(rows)

    train_examples = [
        InputExample(
            texts=[row.anchor, row.positive, row.negative],
            label=1.0               
        )
        for _, row in train_df.iterrows()
    ]

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=32)
    return train_loader

