import os
import tensorflow as tf
import os
import pickle
import numpy as np
import pandas as pd
import librosa
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from utils import extract_features, get_date_string, get_latest_experiment


class CnnModel:
    def __init__(self):
        self.__init__download()
        self.__init__modules()

    def __init__download(self, path="Models"):
        if not os.path.exists(path):
            import gdown

            folder_url = r"https://drive.google.com/drive/folders/1EZL6Ejoa5GH8DzoZcjvRAonlgWznEh14?usp=drive_link"
            gdown.download_folder(folder_url)

    def __init__modules(self, path="Models"):

        print("Loading Prosody Model...")
        self.model = tf.keras.models.load_model(
            os.path.join(path, "Prosody_Model.keras")
        )

        # loding the Scaler
        with open(os.path.join(path, "Prosody_Scaler.pickle"), "rb") as f:
            self.scaler = pickle.load(f)

        # loding the Encoder
        with open(os.path.join(path, "Prosody_Encoder.pickle"), "rb") as f:
            self.encoder = pickle.load(f)

    def get_features(self, audio_path):

        data, sr = librosa.load(
            audio_path, duration=2.5, offset=0
        )  # Extract for 2.5 seconds

        features = extract_features(data)
        features = np.array(features)
        features = np.reshape(features, (1, -1))
        features = self.scaler.transform(features)  # Scaler

        return features

    def predict(self, audio_path):
        features = self.get_features(audio_path)
        prediction = self.model.predict(features)
        y_prediction = self.encoder.inverse_transform(prediction.reshape(1, -1))
        predicted_class = y_prediction[0][0]
        # class probabilities
        predicted_probs = prediction[0]

        # class names prosody_encoder.categories_[0]
        # ['Noise', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        return predicted_class, predicted_probs

    def fine_tuner(self, database_dir):
        print("Starting finetune...")
        data_df = self.create_dataframe(database_dir)
        self.train(data_df)
        print("Finetune finished.")
        
        
    def create_dataframe(self, data_dir: str):

        # todo handle different file formats like mp3
        X = []; Y = []
        for filename in sorted(os.listdir(data_dir)):
            if not filename.endswith(".wav"):
                print(f"Skipping {filename}, not a WAV file.")
                continue
            
            emotion = filename.split("_")[0]
            audio_path = os.path.join(data_dir, filename)

            features = self.get_features(audio_path)[0]
            X.append(features)
            Y.append(emotion)
            
        df = pd.DataFrame(X)
        df["Emotions"] = Y

        return df

    def train(self, data_df: pd.DataFrame, new_experiment=True):
        X = data_df.iloc[:, :-1]
        y = data_df["Emotions"]

        X = self.scaler.transform(X)
        X = np.expand_dims(X, axis=2)
        # todo simplify this complex line
        y = self.encoder.transform(y.to_numpy().reshape(-1, 1)).toarray()

        if new_experiment:
            exp_dir = f"tmp/exp_{get_date_string()}"
        else:
            exp_dir = get_latest_experiment(experiments_dir=r"tmp")

        checkpoint_path = os.path.join(
            exp_dir, r"ckpts/Model_{epoch:02d}-{accuracy:.2f}-{loss:.4f}.keras"
        )

        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
        )
        early_stop = EarlyStopping(
            monitor="loss", mode="auto", patience=5, restore_best_weights=True
        )
        lr_reduction = ReduceLROnPlateau(
            monitor="accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.000001
        )
        optimiser = Adam(learning_rate=1e-3)

        self.model.compile(
            optimizer=optimiser, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        history = self.model.fit(
            X,
            y,
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, lr_reduction, model_checkpoint],
        )

        print("Saving Model Checkpoint...")
        self.model.save(r"./Models/Prosody_Active_Model.keras")

        history_path = exp_dir + "/history.pkl"
        print("Saving History...")
        with open(history_path, "wb") as f:
            pickle.dump(history.history, f)
