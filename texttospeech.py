#Regular Speech Recognition
from speechbrain.inference.ASR import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="pretrained_models/asr-crdnn-rnnlm-librispeech")
asr_model.transcribe_file('TextToSpeech/test3.wav')

signal = read_audio("TextToSpeech/test3.wav").squeeze()
Audio(signal, rate=16000)
