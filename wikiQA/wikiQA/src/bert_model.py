import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)


def train_bert(df, batch_size=2):

    inputs = tokenizer(
        df['Question'].tolist(),
        df['Sentence'].tolist(),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    labels = torch.tensor(df['Label'].values)

    dataset = TensorDataset(
        inputs['input_ids'],
        inputs['attention_mask'],
        labels
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()

    for epoch in range(1):
        total_loss = 0

        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss}")

    return model, tokenizer

def predict_bert(model, tokenizer, question, sentences):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    inputs = tokenizer(
        [question] * len(sentences),
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

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    return ranked