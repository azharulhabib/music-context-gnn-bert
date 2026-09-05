from transformers import BertTokenizer, BertModel
import torch


class BertTextEncoder:
    def __init__(self, model_name="bert-base-uncased", max_length=128, device=None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.max_length = max_length
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        return output.last_hidden_state[:, 0, :]


if __name__ == "__main__":
    encoder = BertTextEncoder()
    sample_texts = ["classical guitar solo", "heavy metal drums", "female opera singing"]
    embeddings = encoder.encode(sample_texts)
    print(f"Device: {encoder.device}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(embeddings[0][:5])