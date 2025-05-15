# testingmodel.py
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = "./final_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

FEWSHOT = (
    "Review: I absolutely love this! It works perfectly.\nSentiment: positive\n"
    "Review: This is the worst purchase I’ve ever made.\nSentiment: negative\n"
    "Review: It’s okay, not great but not terrible.\nSentiment: neutral\n"
)

CANDIDATES = ["positive", "negative", "neutral"]

def score_candidate(prefix: str, candidate: str) -> float:
    prompt_ids    = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1)

    with torch.no_grad():
        logits    = model(full_ids).logits        # (1, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

    # Sum log-probs for each token in candidate
    prompt_len = prompt_ids.shape[1]
    total_lp = 0.0
    for i, tok in enumerate(candidate_ids[0]):
        total_lp += log_probs[0, prompt_len + i - 1, tok].item()
    return total_lp

def predict_sentiment(text: str):
    prefix = FEWSHOT + f"Review: {text}\nSentiment: "
    scores = [score_candidate(prefix, cand) for cand in CANDIDATES]
    probs  = F.softmax(torch.tensor(scores), dim=0).tolist()
    best   = int(torch.argmax(torch.tensor(probs)))
    return CANDIDATES[best], probs[best]

if __name__ == "__main__":
    review = input("Enter a customer review: ")
    sentiment, confidence = predict_sentiment(review)
    print(f"Predicted Sentiment: {sentiment} (confidence {confidence:.2f})")