# 🔍 HSRIS — Hybrid Semantic Retrieval & Intelligence System

> A multi-stage NLP pipeline for processing and retrieving customer support tickets,
> built entirely from scratch using base PyTorch and NumPy.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red)
![GPU](https://img.shields.io/badge/GPU-Dual%20T4-green)
![Dataset](https://img.shields.io/badge/Dataset-8469%20tickets-orange)

---

## 📌 Assignment Info
- **Course:** Data Science for Software Engineering
- **Assignment:** 3 — Hybrid Semantic Retrieval & Intelligence System
- **Platform:** Kaggle (Dual T4 x2 GPU)
- **Dataset:** [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

---

## 🏗️ System Architecture
```
Raw Tickets
    │
    ├──► Label Encoding      (Ticket Priority → ordinal integers)
    ├──► One-Hot Encoding     (Ticket Channel → binary vectors)
    ├──► TF-IDF Pipeline      (sparse tensor, 5000 vocab)
    │       ├── Custom Tokenizer (regex-based)
    │       ├── N-Gram Generator (unigrams + bigrams + trigrams)
    │       └── IDF Computation  (smooth IDF from scratch)
    └──► GloVe Embeddings     (300d, TF-IDF weighted pooling)
            └── OOV → zero vector (<UNK>)

Query
    │
    ├──► TF-IDF Score  (cosine similarity, sparse)
    ├──► GloVe Score   (cosine similarity, dense)
    └──► FinalScore = α × TF-IDF + (1-α) × GloVe
```

---

## 📂 Repository Structure
```
├── DS_ASS01_23F-3079_23F-6032.ipynb   # Main Kaggle notebook
├── app.py                              # Gradio web application
├── benchmark_plot.png                  # GPU execution time plot
├── comparison_plot.png                 # TF-IDF vs GloVe comparison
├── glove_vs_tfidf_qualitative.png      # Qualitative examples plot
├── precision_at_5.png                  # Precision@5 evaluation plot
└── README.md
```

---

## ⚙️ Implementation Details

### Part 1: Categorical Encoders
| Encoder | Field | Output |
|---------|-------|--------|
| Label Encoder | Ticket Priority | (8469,) ordinal integers |
| One-Hot Encoder | Ticket Channel | (8469, 4) binary vectors |

- Built from scratch using Python dicts + PyTorch tensors
- Handles unseen categories during inference
- Priority order: Low=0, Medium=1, High=2, Critical=3

### Part 2: Sparse TF-IDF
- Custom regex tokenizer with lowercasing
- N-gram generation (unigrams, bigrams, trigrams)
- Top 5,000 vocabulary
- Smooth IDF: `log((1+N)/(1+df)) + 1`
- Stored as `torch.sparse` tensor → only 7MB RAM

### Part 3: Dense GloVe Embeddings
- GloVe 300-dimensional vectors (400,001 words)
- Loaded into `torch.nn.Embedding` layer
- TF-IDF weighted mean pooling (prevents semantic dilution)
- OOV tokens → zero vector

---

## 🔍 Hybrid Search Formula
```
FinalScore = α × TF-IDF Score + (1-α) × GloVe Score
```

- Default **α = 0.4** (60% semantic, 40% keyword)
- Adjustable via slider in the deployed app

---

## ⚡ Dual GPU Performance

| Batch Size | Time (ms) | Docs/sec |
|-----------|-----------|----------|
| 1 | 1.12 | 891 |
| 10 | 18.86 | 530 |
| 50 | 19.05 | 2,625 |
| 100 | 19.97 | 5,007 |

- 100 queries processed in **20.23ms**
- **0.202ms** average per query
- `torch.nn.DataParallel` across both T4 GPUs

---

## 📊 Evaluation Results

### Precision@5
| Method | Score |
|--------|-------|
| TF-IDF only (α=1.0) | 15.5% |
| GloVe only (α=0.0) | 17.5% |
| Hybrid (α=0.4) | 15.9% |

### GloVe vs TF-IDF — Qualitative Examples
| Query | Expected | TF-IDF | GloVe |
|-------|----------|--------|-------|
| "I want my money back" | Refund request | ✗ | ✓ |
| "I was charged twice" | Billing inquiry | ✗ | ✓ |
| "I cannot access my account" | Technical issue | ✓ | ✓✓ |
| "I no longer wish to use this" | Cancellation | ✗ | ✓ |
| "Device stopped functioning" | Technical issue | ✓ | ✓ |

---

## 🚀 How to Run

### On Kaggle
1. Open the notebook on Kaggle
2. Set accelerator to **GPU T4 x2**
3. Add the Customer Support Ticket Dataset
4. Click **Run All**

### Gradio App
```bash
pip install gradio
python app.py
```

---

## 🔗 Links
- 📓 **Kaggle Notebook:** https://www.kaggle.com/code/f233079aminanaveed/ds-ass01-23f-3079-23f-6032/
- 📝 **Medium Article:** https://medium.com/@iaminanaveed/building-hsris-a-hybrid-semantic-retrieval-system-from-scratch-with-pytorch-glove-b1515d6cbb2a 
- 💼 **LinkedIn Post:** https://www.linkedin.com/posts/iamna_nlp-machinelearning-pytorch-activity-7440359016440053761-vAR8?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFHnd6MBXjiBaZam5gfrlrwXMJnWpEmkWqs

---

## 👥 Team
|          Name         |
|-----------------------|
| Amina Naveed          |
| Muhammad Bilal        | 

---

## ⚠️ Technical Constraints
- No Scikit-Learn (TfidfVectorizer, LabelEncoder etc.)
- Sparse tensors for TF-IDF (RAM safety)
- Dual GPU via DataParallel
- GloVe OOV handled with zero vector
