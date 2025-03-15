import os
from speechbrain.inference import EncoderClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# CSV_file
test_csv = r'C:\Users\junwe\Desktop\accent_recog\speechbrain\accent-recog-slt2022\CommonAccent\data\test.csv'
data_root = r'C:\Users\junwe\Desktop\accent_recog\speechbrain\accent-recog-slt2022\CommonAccent'
output_folder = 'results/accent-id-commonaccent_ecapa_origin'

# Define all language catagories
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

ACCENT_BREF = [
    'england', 'us', 'canada', 'australia', 'indian', 'scotland', 'ireland', 
    'african', 'malaysia', 'newzealand', 'southatlandtic', 'bermuda', 
    'philippines', 'hongkong', 'wales', 'singapore'
]

selected_accent = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
selected_accent_set = set(selected_accent)

accent_bref = [ACCENT_BREF[i] for i in selected_accent] if selected_accent else ACCENT_BREF

labels = selected_accent if selected_accent else list(range(len(accent_bref)))

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

# Initialize lists to store ground truth and predictions
ground_truth = []
predictions = []

# Iterate over the rows of the DataFrame
for _, row in df.iterrows():
    file_path = row['wav']
    true_accent = accent_2_index[row['accent']]

    if selected_accent_set and (true_accent not in selected_accent_set):
        continue

    # Check if the file exists
    if os.path.exists(file_path):
        # Classify the audio file
        out_scores, max_score, index, text_lab = classifier.classify_file(file_path)

        if selected_accent:
            # Get selected accent scores
            select_scores = out_scores[:, selected_accent]
            
            # Get the index of the highest probability
            max_index = torch.argmax(select_scores, dim=1).item()

            # Map the index back to the accent label
            predicted_accent = selected_accent[max_index]
        else:
            predicted_accent = int(index)

        # Append ground truth and prediction
        ground_truth.append(true_accent)
        predictions.append(predicted_accent)
    else:
        print(f"File not found: {file_path}")

# Compute confusion matrix
conf_matrix = confusion_matrix(ground_truth, predictions, labels=labels)

# Calculate classification metrics
report = classification_report(
    ground_truth,
    predictions,
    labels=labels,
    output_dict=True
)

# Save classification metrics to a CSV file
metrics_df = pd.DataFrame(report).transpose()
metrics_file = os.path.join(output_folder, "classification_metrics.csv")
metrics_df.to_csv(metrics_file, index=True)
print(f"Classification metrics saved to {metrics_file}")

# Save confusion matrix to a CSV file
conf_matrix_df = pd.DataFrame(conf_matrix, index=accent_bref, columns=accent_bref)
conf_matrix_file = os.path.join(output_folder, "confusion_matrix.csv")
conf_matrix_df.to_csv(conf_matrix_file)
print(f"Confusion matrix saved to {conf_matrix_file}")

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Accent')
plt.ylabel('True Accent')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()

# Save confusion matrix plot as an image
conf_matrix_plot_file = os.path.join(output_folder, "confusion_matrix.png")
plt.savefig(conf_matrix_plot_file)
print(f"Confusion matrix plot saved to {conf_matrix_plot_file}")

# Display the plot
plt.show()
