import json, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

TITLES = [
    "Transformer (deep learning)", "Convolutional neural network", "Recurrent neural network",
    "Long short-term memory", "Gradient descent", "Backpropagation", "Random forest",
    "Support vector machine", "K-means clustering", "Principal component analysis",
    "Naive Bayes classifier", "Logistic regression", "Linear regression",
    "Decision tree learning", "Overfitting", "Regularization (mathematics)",
    "Reinforcement learning", "Q-learning", "Generative adversarial network", "Autoencoder",
    "Word embedding", "BERT (language model)", "Attention (machine learning)",
    "Retrieval-augmented generation", "Large language model", "Cross-validation (statistics)",
    "Bias–variance tradeoff", "Gradient boosting", "K-nearest neighbors algorithm",
    "Transfer learning",
]

out = Path("data/docs")
out.mkdir(parents=True, exist_ok=True)
for t in TITLES:
    name = re.sub(r"\W+", "_", t.lower()).strip("_")
    if (out / f"{name}.txt").exists():
        continue
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "redirects": 1, "titles": t, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "rag-eval-student-project/1.0"})
    for attempt in range(5):
        try:
            data = json.load(urllib.request.urlopen(req))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            print("rate limited, waiting 20s...")
            time.sleep(20)
    else:
        print("gave up on", t)
        continue
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract", "")
    if text:
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        print("saved", name)
    else:
        print("missing", t)
    time.sleep(3)