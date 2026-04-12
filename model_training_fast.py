import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback

DATA_DIR = "data"
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
RESIZED_DIR = "data_resized/train"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


labels_df = pd.read_csv(LABELS_FILE)
labels_df["filepath"] = labels_df["id"].apply(lambda x: os.path.join(DATA_DIR, "train", f"{x}.jpg"))


print("Resizing dataset (first run only)...")
os.makedirs(RESIZED_DIR, exist_ok=True)

valid_rows = []
invalid = 0

for idx, row in labels_df.iterrows():
    src = row["filepath"]
    dst = os.path.join(RESIZED_DIR, f"{row['id']}.jpg")

    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize(IMAGE_SIZE, Image.LANCZOS)
            im.save(dst, "JPEG", quality=90)
        valid_rows.append({"id": row["id"], "breed": row["breed"], "filepath": dst})
    except:
        invalid += 1

print(f"Skipped invalid images: {invalid}")

resized_df = pd.DataFrame(valid_rows)


train_df, val_df = train_test_split(
    resized_df, test_size=0.15, stratify=resized_df["breed"], random_state=42
)


train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col="filepath", y_col="breed", target_size=IMAGE_SIZE,
    class_mode="categorical", batch_size=BATCH_SIZE, shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col="filepath", y_col="breed", target_size=IMAGE_SIZE,
    class_mode="categorical", batch_size=BATCH_SIZE, shuffle=False
)

num_classes = len(train_gen.class_indices)


base = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = False  

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"),
                    monitor="val_accuracy", save_best_only=True),
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    TqdmCallback(verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=0
)

model.save(os.path.join(MODEL_DIR, "dog_breed_classifier.h5"))

index_to_class = {v: k for k, v in train_gen.class_indices.items()}
breed_names = [index_to_class[i] for i in range(len(index_to_class))]

np.save(os.path.join(MODEL_DIR, "breed_names.npy"), breed_names)

print("Model saved successfully!")



acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(1, len(acc) + 1) 

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Train Acc")
plt.plot(epochs_range, val_acc, label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Val Loss")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_plot.png"))
plt.close()

print("Training plot saved!")
