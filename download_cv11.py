from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_11_0", "en")

print(next(iter(ds)))