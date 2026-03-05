# brain2speech-model

# 🧠 BRAIN2SPEECH Synthesis - AI Model Pipeline

**Author:** Derin   
**Project Objective:** Analyzing brain signals (iEEG and EEG) and converting them into text and audio outputs using machine learning techniques. 

This repository contains the "Brain" of our capstone project. It takes the processed brain signals from the Data Specialist and runs them through a neural network, then formats the decoded text to be sent to the Backend/Frontend.

---

## 📂 What My Files Do

Right now, the system is running a **"Dummy Pipeline"**. Because the real iEEG data is still being cleaned, this repository uses fake data tensors mapped to the exact shape of Mel-spectrograms so the math can be tested early.

* **`model.py` (The Architecture):** Contains the PyTorch `CNNLSTM_Decoder` class. The CNN learns the spatial electrode patterns, and the LSTM handles the temporal sequence modeling.
* **`dataset.py` (The Bloodstream):** Generates the fake 3D tensors (simulating Mel-spectrogram brain signals) and fake word labels to feed the model.
* **`train.py` (The Heartbeat):** The training loop. It uses **Cross-Entropy Loss** (to grade the model's guesses) and the **Adam Optimizer** (to update the math weights). It saves a `dummy_model.pth` file.
* **`predict.py` (The Handoff):** The offline inference script. It loads the saved `.pth` weights, makes a prediction on a brain signal, and prints a formatted JSON (`text`, `confidence`, `modelVersion`, `audioBase64`) to perfectly match team's API contract.
* **`requirements.txt`:** The libraries needed to run this code (`torch` for the AI, and `mne` for reading the brain files later).

---

## 🛠️ How to Set Up the Environment

If I am opening this project on a new computer or after a long break, here is how I activate my tools:

1. **Open Terminal (PowerShell)** inside the `brain2speech-model` folder.
2. **Create the virtual environment** (if it doesn't exist yet):

   python -m venv venv

   3. **Activate the environment**

   .\venv\Scripts\activate

   (should see (venv) pop up in green on the left side of my terminal)

   4. **Install the required libraries**

   pip install -r requirements.txt

   🚀 How to Run the Code
Step 1: Train the Model
Run this command to push the dummy data through the CNN+LSTM and teach it the patterns.

python train.py

Goal: Watch the loss number go down over the 5 epochs. This will generate the dummy_model.pth file.

Step 2: Run an Offline Prediction
Run this command to simulate a user uploading an EEG file to the website.

Bash
python predict.py
Goal: It should print out a clean JSON string that the FastAPI backend can read.

Next Steps (Phase 2)
When Data Specialist finishes the MNE-Python cleaning pipeline in Month 2, I will:

Delete the fake tensors in dataset.py.

Swap in the real MNE-Python data loader to read her extracted iEEG Mel-spectrograms and labels.

Re-run train.py for real!