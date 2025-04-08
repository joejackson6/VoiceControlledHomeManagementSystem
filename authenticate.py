import os
import torch
import numpy as np
from speechbrain.pretrained import SpeakerRecognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

class VoiceAuthenticator:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.X = []
        self.y = []
        self.label_encoder = LabelEncoder()
        self.classifier = SVC(probability=True)

    def add_user_samples(self, user_name, file_list):
        """
        Adds training samples for a user.
        :param user_name: str
        :param file_list: list of .wav file paths
        """
        for file in file_list:
            emb = self._get_embedding(file)
            self.X.append(emb)
            self.y.append(user_name)

    def train_classifier(self):
        """
        Trains the classifier using added samples.
        """
        if not self.X:
            raise ValueError("No samples provided. Use add_user_samples() first.")
        X = np.array(self.X)
        y = self.label_encoder.fit_transform(self.y)
        self.classifier.fit(X, y)

    def authenticate(self, file_path):
        """
        Authenticates a user based on their voice input.
        :param file_path: str
        :return: str (user name or 'unknown user')
        """
        emb = self._get_embedding(file_path)
        probs = self.classifier.predict_proba([emb])[0]
        max_prob = np.max(probs)
        if max_prob >= self.threshold:
            pred_label = np.argmax(probs)
            user = self.label_encoder.inverse_transform([pred_label])[0]
            return f"Hello, {user}"
        else:
            return "Sorry, unknown user detected."

    def _get_embedding(self, file_path):
        return (
            self.recognizer.encode_batch(file_path)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

    def save(self, path):
        """
        Saves model and label encoder to a directory.
        """
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(path, "classifier.pkl"))
        joblib.dump(self.label_encoder, os.path.join(path, "labels.pkl"))

    def load(self, path):
        """
        Loads model and label encoder from a directory.
        """
        self.classifier = joblib.load(os.path.join(path, "classifier.pkl"))
        self.label_encoder = joblib.load(os.path.join(path, "labels.pkl"))


def load_voice_data_from_folder(authenticator, base_path="VoiceData"):
    """
    Loads all WAV files from each user’s subdirectory.
    :param authenticator: VoiceAuthenticator instance
    :param base_path: path to VoiceData folder
    """
    for user in os.listdir(base_path):
        user_path = os.path.join(base_path, user)
        if os.path.isdir(user_path):
            wav_files = [
                os.path.join(user_path, f)
                for f in os.listdir(user_path)
                if f.lower().endswith(".wav")
            ]
            if wav_files:
                print(f"Adding {len(wav_files)} files for user: {user}")
                authenticator.add_user_samples(user, wav_files)


if __name__ == "__main__":
    # === Step 1: Create instance
    auth = VoiceAuthenticator(threshold=0.7)

    # === Step 2: Load user data from folders
    load_voice_data_from_folder(auth, base_path="VoiceData")

    # === Step 3: Train the model
    print("Training voice classifier...")
    auth.train_classifier()

    # === Step 4: Authenticate a test sample
    test_file = "test_input.wav"  # Replace with your test file path
    print("Authenticating test input...")
    result = auth.authenticate(test_file)
    print(result)

