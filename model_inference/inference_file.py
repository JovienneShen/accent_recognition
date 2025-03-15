import os
import glob
import torchaudio
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from pydub import AudioSegment
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Define all language catagories
languages = {
    0: 'england', 1: 'us', 2: 'canada', 3: 'australia', 4: 'indian', 
    5: 'scotland', 6: 'ireland', 7: 'african', 8: 'malaysia', 
    9: 'newzealand', 10: 'southatlandtic', 11: 'bermuda', 
    12: 'philippines', 13: 'hongkong', 14: 'wales', 15: 'singapore'
}

select_languages = [0, 1]

# Initialize the classifier
classifier = EncoderClassifier.from_hparams(
    # source="Jzuluaga/accent-id-commonaccent_ecapa", 
    # savedir="pretrained_models/accent-id-commonaccent_ecapa"
    source="pretrained_models/ECAPA-TDNN-241229-70k-filtered-0.7/1986/save",
    savedir="pretrained_models/ECAPA-TDNN-241229-70k-filtered-0.7/1986/save",
)

# Get all labels
all_labels = classifier.hparams.label_encoder.ind2lab
print()
print("All categories:", all_labels)

# Define input and output directories
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir_name = "audios"
input_dir = os.path.join(script_dir, input_dir_name)       # e.g., ./input_audio/
output_dir = os.path.join(script_dir, "outputs")   # e.g., ./output_results/
output_csv = os.path.join(output_dir, "accent_detection_results.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to store results
results = []

# Define supported audio formats
supported_formats = ['.wav']

# Iterate over all supported audio files in the input directory
for ext in supported_formats:
    files = []
    for root, dirs, files_in_dir in os.walk(input_dir):
        for fname in files_in_dir:
            if fname.lower().endswith(ext):
                files.append(os.path.join(root, fname))

    print(f"Found {len(files)} {ext} files to process.")
    
    for audio_path in tqdm(files):

        file_name = os.path.basename(audio_path)

        if 'francais' in audio_path:
            continue

        audio_path = os.path.abspath(audio_path)
        audio_path = Path(audio_path).as_posix()

        # Classify the audio file
        out_scores, max_score, index, text_lab = classifier.classify_file(audio_path)
        
        # Get selected accent scores
        select_scores = out_scores[:, select_languages]

        # probabilities = torch.exp(out_scores)
        # normalized_probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        # # Get the predicted label
        # predicted_label = text_lab[0]
        # # Get the probability of the predicted label
        # predicted_prob = normalized_probabilities[0, index].item()
        
        # Calculate normalized probabilities
        select_probs = (torch.exp(select_scores) / torch.exp(select_scores).sum())[0]
        
        # Get the index of the highest probability
        max_index = torch.argmax(select_scores, dim=1).item()

        # Map index to label
        max_lan_id = select_languages[max_index]
        predict_acc = languages[max_lan_id]

        # construct result dict
        prob_dict = {
            "filename": os.path.basename(file_name),
            "predicted_accent": predict_acc
        }
        for i, lan_id in enumerate(select_languages):
            prob_dict[languages[lan_id]] = select_probs[i].item()
            
        # Append the result
        results.append(prob_dict)
        
        # print(f"Processed {os.path.basename(file_path)}: {predict_acc} ({select_probs[max_index]})")

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Save the results to a CSV file
df.to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
