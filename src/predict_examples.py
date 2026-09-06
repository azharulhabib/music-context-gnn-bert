import torch
from train import BertTagClassifier, load_musiccaps, build_tag_vocab, build_labels
from transformers import BertTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

df = load_musiccaps()
vocab = build_tag_vocab(df)
model = BertTagClassifier(n_tags=len(vocab)).to(DEVICE)
model.load_state_dict(torch.load("results/bert_tag_classifier.pt"))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for i in range(5):
    caption = df.iloc[i]["caption"]
    true_tags = [t for t in df.iloc[i]["aspects"] if t in vocab]

    tokens = tokenizer(caption, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(tokens["input_ids"], tokens["attention_mask"])
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    pred_tags = [vocab[j] for j in range(len(vocab)) if probs[j] > 0.5]

    print(f"\nCaption: {caption[:80]}...")
    print(f"True tags: {true_tags}")
    print(f"Predicted tags: {pred_tags}")