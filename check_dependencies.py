import subprocess
import sys

required = [
    "torch",
    "torchaudio",
    "speechbrain",
    "sounddevice",
    "soundfile",
    "scikit-learn",
    "pyttsx3",
    "SpeechRecognition",
    "pytz",
    "yeelight"

]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in required:
    try:
        __import__(pkg)
        print(f"{pkg} is installed.")
    except ImportError:
        print(f"{pkg} is missing. Installing...")
        install(pkg)
