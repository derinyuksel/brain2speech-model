import json
import torch
from model import CNNLSTM_Decoder

def run_inference(file_path):
    # 1. Load the blank architecture
    model = CNNLSTM_Decoder(input_channels=64, cnn_hidden=128, lstm_hidden=256, num_classes=50)
    
    # 2. Load the brain weights we just trained in train.py
    try:
        model.load_state_dict(torch.load("dummy_model.pth"))
        # Put model in evaluation mode (turns off training-specific stuff)
        model.eval() 
    except FileNotFoundError:
        return json.dumps({"error": "Model file missing. Run train.py first!"})

    # 3. Simulate Razan's MNE-Python preprocessing.
    # When Razan finishes her pipeline, we will replace this dummy tensor 
    # with the actual loaded .edf or .mat file features.
    # Shape: (batch_size=1, channels=64, time_sequence=100)
    dummy_signal = torch.randn(1, 64, 100)

    # 4. Run the signal through the model
    # no_grad() saves memory because we don't need to learn anything here
    with torch.no_grad(): 
        output = model(dummy_signal)
        
        # Calculate confidence score and grab the highest probability guess
        probabilities = torch.softmax(output, dim=1)
        confidence_score, prediction_index = torch.max(probabilities, dim=1)

    # Fake vocabulary mapping for now
    word = f"decoded_word_{prediction_index.item()}"
    confidence = round(confidence_score.item(), 2)

    # 5. Format the output to exactly match Sudenur's API Contract
    result = {
        "text": word,
        "confidence": confidence,
        "modelVersion": "dummy-v1.0",
        "audioBase64": None # Keeping None until we add the Vocoder in Phase 4
    }

    # Print it so the backend can read the JSON string
    print(json.dumps(result))
    return json.dumps(result)

if __name__ == "__main__":
    # Simulating Sudenur's backend handing us an uploaded user file
    run_inference("test_upload.edf")