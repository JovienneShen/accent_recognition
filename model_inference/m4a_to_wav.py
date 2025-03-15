import torchaudio
from speechbrain.inference import EncoderClassifier
from pydub import AudioSegment
import numpy as np
import torch

classifier = EncoderClassifier.from_hparams(source="Jzuluaga/accent-id-commonaccent_ecapa", savedir="pretrained_models/accent-id-commonaccent_ecapa")

# Get all labels
all_labels = classifier.hparams.label_encoder.ind2lab
print("All categories:", all_labels)

filename = 'indian1_clip1'
audio = AudioSegment.from_file(f"{filename}.m4a", format="m4a")
audio.export(f"{filename}.wav", format="wav")

out_prob, score, index, text_lab = classifier.classify_file(f'{filename}.wav')

print("Text label = ", text_lab)
probabilities = torch.exp(out_prob)
normalized_probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

print(normalized_probabilities)

eng_us_prob = probabilities[:, [0, 1]]
print(eng_us_prob, eng_us_prob / eng_us_prob.sum())