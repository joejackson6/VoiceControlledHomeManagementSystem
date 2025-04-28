# VoiceControlledHomeManagementSystem

# Dependencies

# Step 1: Install python3-venv if it's missing
sudo apt update
sudo apt install python3-venv

# Step 2: Create a virtual environment
python3 -m venv venv

# Step 3: Activate the venv
source venv/bin/activate

# Step 4: Upgrade pip inside the venv
pip install --upgrade pip

# Step 5: 
sudo apt update
sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev

# Step 6:
python check_dependencies.py
