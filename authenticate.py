import os
import torch
import torchaudio
import numpy as np
from speechbrain.inference import SpeakerRecognition
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
        for file in file_list:
            emb = self._get_embedding(file)
            self.X.append(emb)
            self.y.append(user_name)

    def train_classifier(self):
        if not self.X:
            raise ValueError("No samples provided. Use add_user_samples() first.")
        X = np.array(self.X)
        y = self.label_encoder.fit_transform(self.y)
        self.classifier.fit(X, y)

    def authenticate(self, file_path):
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
        waveform, sample_rate = torchaudio.load(file_path)
        return (
            self.recognizer.encode_batch(waveform)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(path, "classifier.pkl"))
        joblib.dump(self.label_encoder, os.path.join(path, "labels.pkl"))

    def load(self, path):
        self.classifier = joblib.load(os.path.join(path, "classifier.pkl"))
        self.label_encoder = joblib.load(os.path.join(path, "labels.pkl"))

def load_voice_data_from_folder(authenticator, base_path="VoiceData"):
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

class VoiceAuthenticatorFromEmbeddings:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        self.classifier = SVC(probability=True)
        self.label_encoder = LabelEncoder()

    def train_from_pt(self, path="embeddings.pt"):
        data = torch.load(path)
        X = []
        y = []
        for user, emb in data.items():
            #X.append(emb.cpu().numpy())
            X.append(emb.cpu().numpy().flatten())
            y.append(user)
        self.X = np.array(X)
        self.y = self.label_encoder.fit_transform(y)
        self.classifier.fit(self.X, self.y)

    def authenticate(self, wav_path):
        waveform, sample_rate = torchaudio.load(wav_path)
        emb = (
            self.recognizer.encode_batch(waveform)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )
        probs = self.classifier.predict_proba([emb])[0]
        labels = self.label_encoder.inverse_transform(np.arange(len(probs)))

        for label, score in zip(labels, probs):
            print(f"{label}: {score:.4f}")

        max_prob = np.max(probs)
        if max_prob >= self.threshold:
            label = np.argmax(probs)
            user = self.label_encoder.inverse_transform([label])[0]
            return f"Authenticated as {user} (confidence: {max_prob:.4f})"
        else:
            return f"Unknown user (highest confidence: {max_prob:.4f})"

    
if __name__ == "__main__":
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import torchaudio
    import numpy as np
    import torch

    devices = sd.query_devices()
    k66_index = None
    for i, d in enumerate(devices):
        if "K66" in d['name']:
            k66_index = i
            break

    if k66_index is None:
        print("Error: No input device found with 'K66' in name.")
        exit(1)

    raw_rate = 44100
    target_rate = 16000
    duration = 5
    print("Start speaking now...")
    recording = sd.rec(int(duration * raw_rate), samplerate=raw_rate, channels=1, dtype='float32', device=k66_index)
    sd.wait()
    print("Recording finished.")

    recording = torch.tensor(recording).transpose(0, 1)
    resampled = torchaudio.transforms.Resample(orig_freq=raw_rate, new_freq=target_rate)(recording)
    torchaudio.save("test_input.wav", resampled, target_rate)

    auth = VoiceAuthenticatorFromEmbeddings(threshold=0.7)
    auth.train_from_pt("embeddings.pt")
    print(auth.authenticate("test_input.wav"))

