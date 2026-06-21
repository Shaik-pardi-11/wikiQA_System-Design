from src.preprocessing import load_data, clean_data
from src.model import train_model, predict
from src.evaluate import compute_map, compute_mrr
from src.bert_model import train_bert, predict_bert


df = load_data("data/train.tsv")
df = clean_data(df)

# ------------------ BASELINE MODEL ------------------
model, vectorizer = train_model(df)

sample_q = df['Question'].iloc[0]
subset = df[df['Question'] == sample_q]
sentences = subset['Sentence'].tolist()

results = predict(model, vectorizer, sample_q, sentences)

print("\nTop Answers (TF-IDF):\n")
for s, score in results[:5]:
    print(score, "->", s)

# Evaluation
grouped = df.groupby('Question')

mrr = compute_mrr(grouped, model, vectorizer)
map_score = compute_map(grouped, model, vectorizer)

print("\nMRR:", mrr)
print("MAP:", map_score)

# ------------------ BERT MODEL ------------------
# Reduce dataset size (VERY IMPORTANT)
bert_df = df.sample(300, random_state=42)

bert_model, tokenizer = train_bert(bert_df, batch_size=2)

results = predict_bert(bert_model, tokenizer, sample_q, sentences)

print("\nBERT Top Answers:\n")
for s, score in results[:5]:
    print(score, "->", s)

from src.evaluate import compute_mrr_bert, compute_map_bert

grouped_bert = bert_df.groupby('Question')

bert_mrr = compute_mrr_bert(grouped_bert, bert_model, tokenizer)
bert_map = compute_map_bert(grouped_bert, bert_model, tokenizer)

print("\nBERT MRR:", bert_mrr)
print("BERT MAP:", bert_map)