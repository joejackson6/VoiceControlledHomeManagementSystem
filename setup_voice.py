import os
from pathlib import Path
import torch
import torchaudio
from speechbrain.inference import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
	source="speechbrain/spkrec-ecapa-voxceleb",
	savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

base_path = Path("VoiceData")
user_dirs = [d for d in base_path.iterdir() if d.is_dir()]
user_embeddings = {}

for user_dir in user_dirs:
	print(f"Scanning user: {user_dir.name}")
	wav_files = list(user_dir.glob("*.wav"))
	print(f"  Found {len(wav_files)} .wav files")

	if not wav_files:
		continue

	embeddings = []
	for wav in wav_files:
		try:
			waveform, sample_rate = torchaudio.load(str(wav))
			embedding = verification.encode_batch(waveform).squeeze(0).detach()
			embeddings.append(embedding)
			print(f"  Processed: {wav.name}")
		except Exception as e:
			print(f"  Failed to process {wav.name}: {e}")

	if embeddings:
		avg_embedding = torch.stack(embeddings).mean(dim=0)
		user_embeddings[user_dir.name] = avg_embedding
		print(f"  Stored averaged embedding for: {user_dir.name}")

torch.save(user_embeddings, "embeddings.pt")
print(f"\nDone. Stored {len(user_embeddings)} averaged embeddings in embeddings.pt")

