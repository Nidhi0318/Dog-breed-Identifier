import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
RESIZED_DIR = "data_resized/train"


labels_df = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

rows = []
missing = 0

for i, row in labels_df.iterrows():
    img_id = row["id"]
    breed = row["breed"]
    resized_path = os.path.join(RESIZED_DIR, f"{img_id}.jpg")

    if os.path.exists(resized_path):
        rows.append({"id": img_id, "breed": breed, "filepath": resized_path})
    else:
        missing += 1

df = pd.DataFrame(rows)

df.to_csv(os.path.join(RESIZED_DIR, "resized_labels.csv"), index=False)

print("resized_labels.csv generated successfully!")
print("Missing resized images:", missing)
print("Total valid resized images:", len(df))

