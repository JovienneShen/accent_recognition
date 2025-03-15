import os
from speechbrain.inference import EncoderClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# CSV file
test_csv = r'C:\Users\junwe\Desktop\accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\accent_id\data\en_us_70k\train.csv'
data_root = r'C:\Users\junwe\Desktop\accent_recog\speechbrain\accent-recog-slt2022\CommonAccent'
output_folder = 'results/accent-id-commonaccent_ecapa_origin_data_filtered_from_70k'
filtered_csv_file = os.path.join(output_folder, "filtered_data.csv")

# Define all language categories
accent_2_index = {
    'ENGLAND ENGLISH': 0, 
    'UNITED STATES ENGLISH': 1, 
    'CANADIAN ENGLISH': 2, 
    'AUSTRALIAN ENGLISH': 3, 
    'INDIA AND SOUTH ASIA INDIA PAKISTAN SRI LANKA': 4, 
    'SCOTTISH ENGLISH': 5, 
    'IRISH ENGLISH': 6, 
    'SOUTHERN AFRICAN SOUTH AFRICA ZIMBABWE NAMIBIA': 7, 
    'MALAYSIAN ENGLISH': 8, 
    'NEW ZEALAND ENGLISH': 9, 
    'southatlandtic': 10, 
    'WEST INDIES AND BERMUDA BAHAMAS BERMUDA JAMAICA TRINIDAD': 11, 
    'FILIPINO': 12, 
    'HONG KONG ENGLISH': 13, 
    'WELSH ENGLISH': 14, 
    'SINGAPOREAN ENGLISH': 15
}

# Threshold for filtering
THRESHOLD = 0.7  # Adjust as needed

# Initialize the classifier
classifier = EncoderClassifier.from_hparams(
    source="Jzuluaga/accent-id-commonaccent_ecapa", 
    savedir="pretrained_models/accent_id_commonaccent_ecapa"
)

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

# Load the CSV file
df = pd.read_csv(test_csv)

# Replace $data_root with the actual data root path in the 'wav' column
df['wav'] = df['wav'].str.replace('$data_root', data_root)

# Initialize lists to store filtered data
filtered_data = []

# Iterate over the rows of the DataFrame
for _, row in df.iterrows():
    file_path = row['wav']
    true_accent = accent_2_index[row['accent']]

    # Check if the file exists
    if os.path.exists(file_path):
        # Classify the audio file
        out_scores, max_score, index, text_lab = classifier.classify_file(file_path)

        # Get probability of the correct accent
        true_accent_prob = out_scores[0, true_accent].item()

        # If probability is higher than threshold, add to filtered data
        if true_accent_prob >= THRESHOLD:
            filtered_data.append(row)

    else:
        print(f"File not found: {file_path}")

# Convert filtered data to DataFrame and save
filtered_df = pd.DataFrame(filtered_data)
filtered_df.to_csv(filtered_csv_file, index=False)
print(f"Filtered data containing {len(filtered_data)} examples saved to {filtered_csv_file}.")
