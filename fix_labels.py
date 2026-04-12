import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

df = pd.read_csv("data_resized/train/resized_labels.csv")

datagen = ImageDataGenerator()
gen = datagen.flow_from_dataframe(
    df, x_col="filepath", y_col="breed",
    class_mode="categorical", target_size=(224,224)
)

index_to_class = {v: k for k, v in gen.class_indices.items()}
breed_names = [index_to_class[i] for i in range(len(index_to_class))]

np.save("models/breed_names.npy", breed_names)

print("Fixed breed_names.npy saved!")
 