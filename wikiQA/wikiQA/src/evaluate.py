import numpy as np
import torch

# =========================================================
# 🔹 TF-IDF / Logistic Regression Evaluation
# =========================================================

def compute_mrr(grouped_data, model, vectorizer):
    reciprocal_ranks = []

    for q, group in grouped_data:
        sentences = group['Sentence'].tolist()
        labels = group['Label'].tolist()

        inputs = [q + " " + s for s in sentences]
        X = vectorizer.transform(inputs)
        scores = model.predict_proba(X)[:, 1]

        ranked = sorted(zip(scores, labels), reverse=True)

        for i, (_, label) in enumerate(ranked):
            if label == 1:
                reciprocal_ranks.append(1 / (i + 1))
                break

    return np.mean(reciprocal_ranks)


def compute_map(grouped_data, model, vectorizer):
    average_precisions = []

    for q, group in grouped_data:
        sentences = group['Sentence'].tolist()
        labels = group['Label'].tolist()

        inputs = [q + " " + s for s in sentences]
        X = vectorizer.transform(inputs)
        scores = model.predict_proba(X)[:, 1]

        ranked = sorted(zip(scores, labels), reverse=True)

        num_correct = 0
        precision_sum = 0

        for i, (_, label) in enumerate(ranked):
            if label == 1:
                num_correct += 1
                precision_sum += num_correct / (i + 1)

        if num_correct > 0:
            average_precisions.append(precision_sum / num_correct)

    return np.mean(average_precisions)


# =========================================================
# 🔹 BERT Evaluation
# =========================================================

def compute_mrr_bert(grouped_data, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    reciprocal_ranks = []

    for q, group in grouped_data:
        sentences = group['Sentence'].tolist()
        labels = group['Label'].tolist()

        inputs = tokenizer(
            [q] * len(sentences),
            sentences,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        ranked = sorted(zip(scores, labels), reverse=True)

        for i, (_, label) in enumerate(ranked):
            if label == 1:
                reciprocal_ranks.append(1 / (i + 1))
                break

    return np.mean(reciprocal_ranks)


def compute_map_bert(grouped_data, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    average_precisions = []

    for q, group in grouped_data:
        sentences = group['Sentence'].tolist()
        labels = group['Label'].tolist()

        inputs = tokenizer(
            [q] * len(sentences),
            sentences,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        ranked = sorted(zip(scores, labels), reverse=True)

        num_correct = 0
        precision_sum = 0

        for i, (_, label) in enumerate(ranked):
            if label == 1:
                num_correct += 1
                precision_sum += num_correct / (i + 1)

        if num_correct > 0:
            average_precisions.append(precision_sum / num_correct)

    return np.mean(average_precisions)