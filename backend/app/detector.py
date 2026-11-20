import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "roberta"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Classification threshold
# Can be overridden using the SENTINELAI_THRESHOLD environment variable.
CLASSIFICATION_THRESHOLD = float(
    os.getenv("SENTINELAI_THRESHOLD", "0.20")
)

if not 0.0 <= CLASSIFICATION_THRESHOLD <= 1.0:
    raise ValueError(
        "SENTINELAI_THRESHOLD must be between 0.0 and 1.0"
    )


class SentinelDetector:

    def __init__(self):

        print("Loading SentinelAI RoBERTa model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR
        )

        self.model.to(DEVICE)
        self.model.eval()

        self.threshold = CLASSIFICATION_THRESHOLD

        print("RoBERTa loaded successfully!")
        print(f"Device: {DEVICE}")
        print(f"Classification threshold: {self.threshold}")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

        benign_probability = probabilities[0].item()
        malicious_probability = probabilities[1].item()

        label = (
            "MALICIOUS"
            if malicious_probability >= self.threshold
            else "BENIGN"
        )

        return {
            "prediction": label,
            "benign_probability": round(
                benign_probability * 100, 2
            ),
            "malicious_probability": round(
                malicious_probability * 100, 2
            )
        }


detector = SentinelDetector()