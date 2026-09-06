import pandas as pd
from sklearn.model_selection import train_test_split

annotations = pd.read_csv("data/raw/magnatagatune_annotations.csv", sep="\t")
train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=42)

train_df[["clip_id"]].to_csv("data/splits/train_ids.csv", index=False)
val_df[["clip_id"]].to_csv("data/splits/val_ids.csv", index=False)

print(f"Train: {len(train_df)}, Val: {len(val_df)}")