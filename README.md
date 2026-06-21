# WikiQA - Question Answering System

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-red)]()

A hybrid Question Answering system combining **TF-IDF with Logistic Regression** and **BERT-based deep learning** approaches to retrieve and rank relevant answers from a dataset of Q&A pairs.

##  Features

-  **Dual Model Approach**: TF-IDF baseline + BERT fine-tuning
-  **Interactive Web UI**: Streamlit-based interface
-  **Evaluation Metrics**: MRR (Mean Reciprocal Rank) & MAP (Mean Average Precision)
-  **GPU Support**: Accelerated BERT inference with CUDA
-  **Easy to Deploy**: Streamlit Cloud ready
-  **Reproducible**: Fixed random seeds for consistency

##  Requirements

### System Requirements
- **Python**: 3.8 - 3.11
- **RAM**: 8 GB minimum (16 GB recommended)
- **Disk**: 5 GB minimum
- **GPU**: Optional (NVIDIA GPU with CUDA for faster BERT training)

### Python Dependencies
```bash
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
torch>=1.9.0
transformers>=4.6.0
streamlit>=1.0.0
```

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/wikiQA.git
cd wikiQA
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application

**Option A: Command-line Training & Evaluation**
```bash
python src/main.py
```

**Option B: Interactive Web Application**
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

##  Project Structure

```
wikiQA/
├── app.py                  # Streamlit web application
├── src/
│   ├── __init__.py
│   ├── model.py           # TF-IDF + Logistic Regression
│   ├── bert_model.py      # BERT model implementation
│   ├── preprocessing.py   # Data loading & cleaning
│   ├── evaluate.py        # Evaluation metrics (MRR, MAP)
│   └── main.py            # Training pipeline
├── data/
│   ├── train.tsv          # Training data
│   ├── dev.tsv            # Development data
│   └── test.tsv           # Test data
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

##  Data Format

The project expects TSV (Tab-Separated Values) files with three columns:

| Column | Type | Description |
|--------|------|-------------|
| Question | string | The query text |
| Sentence | string | The candidate answer |
| Label | int | 1 (relevant), 0 (non-relevant) |

Example:
```
What is Python?	Python is a programming language	1
What is Python?	Paris is a city in France	0
```

##  Models

### Model 1: TF-IDF + Logistic Regression
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifier**: Logistic Regression
- **Training**: Full dataset
- **Inference**: Fast
- **Expected MRR**: 0.3-0.5

### Model 2: BERT
- **Base Model**: `bert-base-uncased`
- **Task**: Sequence Classification
- **Fine-tuning**: On sampled subset (300 samples)
- **Optimization**: AdamW (lr=2e-5)
- **Batch Size**: 2
- **Inference**: GPU-accelerated
- **Expected MRR**: 0.5-0.7+

![Sample Screenshot](screenshot.png)

##  Evaluation Metrics

### Mean Reciprocal Rank (MRR)
Measures how high the first relevant answer is ranked.
```
MRR = (1/|Q|) * Σ (1 / rank_of_first_relevant)
```

### Mean Average Precision (MAP)
Measures precision at each position where a relevant answer appears.
```
MAP = (1/|Q|) * Σ (1/num_relevant) * Σ (precision@k)
```

##  Web Interface

The Streamlit app provides an interactive interface with:
- Text input for questions
- Real-time model inference
- Results from both TF-IDF and BERT models
- Side-by-side comparison of approaches


##  Configuration

Key hyperparameters (adjustable in source code):

```python
# In src/bert_model.py
batch_size = 2              # Reduce if experiencing OOM
max_length = 64             # Maximum token length
learning_rate = 2e-5        # BERT fine-tuning learning rate
epochs = 1                  # Training epochs (increase for better results)

# In app.py
sample_size = 300           # Number of examples for BERT training
random_state = 42           # For reproducibility
```

##  Usage Examples

### Command Line Usage
```python
from src.preprocessing import load_data, clean_data
from src.model import train_model, predict

# Load and prepare data
df = load_data("data/train.tsv")
df = clean_data(df)

# Train TF-IDF model
model, vectorizer = train_model(df)

# Make predictions
question = "What is machine learning?"
sentences = ["ML is a subset of AI", "Python is a language"]
results = predict(model, vectorizer, question, sentences)

# Results: [(sentence, score), ...]
for sentence, score in results[:5]:
    print(f"{score:.3f} → {sentence}")
```

### Web App Usage
1. Navigate to http://localhost:8501
2. Enter a question in the text box
3. Click "Get Answers"
4. View results from both models

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy automatically

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t wikiqa .
docker run -p 8501:8501 wikiqa
```

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| CUDA out of memory | Reduce `batch_size` or use CPU |
| Data files not found | Ensure TSV files are in `data/` directory |
| BERT download fails | Check internet connection |
| Port 8501 already in use | Use `streamlit run app.py --server.port 8502` |

##  Performance Benchmarks

On a machine with 16GB RAM and an NVIDIA GPU:

| Model | Training Time | Inference Time | MRR | MAP |
|-------|--------------|----------------|-----|-----|
| TF-IDF | <1 second | <10ms | 0.42 | 0.31 |
| BERT | ~2 minutes | ~50ms | 0.65 | 0.54 |

##  Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
- [scikit-learn](https://scikit-learn.org)
- [WikiQA Dataset Paper](https://aclanthology.org/D15-1237/)

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



---

 If you found this helpful, please give it a star!
