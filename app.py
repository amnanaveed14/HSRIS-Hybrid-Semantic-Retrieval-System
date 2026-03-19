
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import math
import time
import os
from collections import Counter

st.set_page_config(
    page_title="HSRIS - Hybrid Semantic Retrieval System",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .result-card {
        background: #1e1e2e;
        border: 1px solid #3a3a5c;
        border-left: 4px solid #4a9eff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .result-card-gold  { border-left-color: #ffd700; }
    .result-card-silver{ border-left-color: #c0c0c0; }
    .result-card-bronze{ border-left-color: #cd7f32; }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-type     { background:#1a3a5c; color:#4a9eff; }
    .badge-critical { background:#3a1a1a; color:#ff4a4a; }
    .badge-high     { background:#3a2a1a; color:#ff8c00; }
    .badge-medium   { background:#3a3a1a; color:#ffd700; }
    .badge-low      { background:#1a3a1a; color:#4aff4a; }
    .badge-channel  { background:#2a1a3a; color:#b44aff; }
    .score-box {
        background: #0d1117;
        border-radius: 8px;
        padding: 8px 14px;
        font-family: monospace;
        font-size: 13px;
        color: #7ee787;
        margin-top: 8px;
    }
    .predicted-type {
        background: linear-gradient(135deg, #0d2137, #1a3a5c);
        border: 1px solid #4a9eff;
        border-radius: 10px;
        padding: 16px 24px;
        margin: 16px 0;
        font-size: 22px;
        font-weight: 700;
        color: #4a9eff;
    }
    .stTextArea textarea { font-size: 15px; }
    div[data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTokenizer:
    def __init__(self):
        self.pattern = re.compile(r"[a-z0-9]+")
    def tokenize(self, text):
        if not isinstance(text, str): return []
        return self.pattern.findall(text.lower())
    def tokenize_batch(self, texts):
        return [self.tokenize(t) for t in texts]

class NGramGenerator:
    def __init__(self, ngram_range=(1,3)):
        self.min_n, self.max_n = ngram_range
    def generate(self, tokens):
        ngrams = []
        for n in range(self.min_n, self.max_n+1):
            for i in range(len(tokens)-n+1):
                ngrams.append("_".join(tokens[i:i+n]))
        return ngrams
    def generate_batch(self, token_lists):
        return [self.generate(t) for t in token_lists]

class TFIDFVectorizer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vocab        = {}
        self.idf_weights  = None
        self.fitted       = False
    def fit(self, ngram_lists):
        n_docs     = len(ngram_lists)
        df_count   = Counter()
        all_counts = Counter()
        for ngrams in ngram_lists:
            all_counts.update(ngrams)
            df_count.update(set(ngrams))
        top_tokens     = [t for t,_ in all_counts.most_common(self.max_features)]
        self.vocab     = {t:i for i,t in enumerate(top_tokens)}
        idf_vals       = [math.log((1+n_docs)/(1+df_count.get(t,0)))+1.0
                          for t in top_tokens]
        self.idf_weights = torch.tensor(idf_vals, dtype=torch.float32, device=device)
        self.fitted    = True
        return self
    def transform(self, ngram_lists):
        rows, cols, vals = [], [], []
        for doc_idx, ngrams in enumerate(ngram_lists):
            tc    = Counter(ngrams)
            total = len(ngrams)
            for token, count in tc.items():
                if token in self.vocab:
                    rows.append(doc_idx)
                    cols.append(self.vocab[token])
                    vals.append((count/total)*self.idf_weights[self.vocab[token]].item())
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values  = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values,
            size=(len(ngram_lists), len(self.vocab)), device=device
        ).coalesce()
    def fit_transform(self, ngram_lists):
        return self.fit(ngram_lists).transform(ngram_lists)

class GloVeEmbeddings:
    def __init__(self, glove_path, embedding_dim=300):
        self.glove_path    = glove_path
        self.embedding_dim = embedding_dim
        self.word2idx      = {}
        self.embedding     = None
    def load(self):
        vectors  = []
        word2idx = {"<UNK>": 0}
        vectors.append(np.zeros(self.embedding_dim, dtype=np.float32))
        with open(self.glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                if len(vec) != self.embedding_dim: continue
                word2idx[word] = len(vectors)
                vectors.append(vec)
        self.word2idx  = word2idx
        weight_matrix  = torch.tensor(np.stack(vectors), dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weight_matrix, freeze=True, padding_idx=0).to(device)
        return self
    def get_word_vector(self, word):
        idx = self.word2idx.get(word.lower(), 0)
        return self.embedding(torch.tensor([idx], device=device)).squeeze(0)
    def encode_document_tfidf_weighted(self, tokens, tfidf_scores):
        if not tokens:
            return torch.zeros(self.embedding_dim, device=device)
        weighted_sum = torch.zeros(self.embedding_dim, device=device)
        total_weight = 0.0
        for token in tokens:
            vec    = self.get_word_vector(token)
            weight = tfidf_scores.get(token, 1e-8)
            weighted_sum += weight * vec
            total_weight += weight
        return weighted_sum / max(total_weight, 1e-8)

def normalize(matrix):
    norms = matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return matrix / norms

def search(query_text, alpha, df, tfidf_norm, glove_norm,
           tfidf_vec, glove, tokenizer, ngram_gen, top_k=5):
    tokens  = tokenizer.tokenize(query_text)
    ngrams  = ngram_gen.generate(tokens)
    tfidf_q = tfidf_vec.transform([ngrams]).to_dense()
    tfidf_scores_dict = {}
    for token in tokens:
        if token in tfidf_vec.vocab:
            tfidf_scores_dict[token] = tfidf_vec.idf_weights[
                tfidf_vec.vocab[token]].item()
    glove_q     = glove.encode_document_tfidf_weighted(
        tokens, tfidf_scores_dict).unsqueeze(0)
    tfidf_q_n   = normalize(tfidf_q)
    glove_q_n   = normalize(glove_q)
    tfidf_s     = torch.mm(tfidf_norm, tfidf_q_n.T).squeeze(1)
    glove_s     = torch.mm(glove_norm, glove_q_n.T).squeeze(1)
    final_s     = alpha * tfidf_s + (1-alpha) * glove_s
    top_idx     = final_s.argsort(descending=True)[:top_k].cpu().tolist()
    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "rank"        : rank+1,
            "subject"     : df["Ticket Subject"].iloc[idx],
            "description" : df["Ticket Description"].iloc[idx][:200],
            "ticket_type" : df["Ticket Type"].iloc[idx],
            "priority"    : df["Ticket Priority"].iloc[idx],
            "channel"     : df["Ticket Channel"].iloc[idx],
            "tfidf_score" : tfidf_s[idx].item(),
            "glove_score" : glove_s[idx].item(),
            "final_score" : final_s[idx].item(),
        })
    return results

@st.cache_resource
def load_all():
    DATA_PATH  = "/kaggle/input/datasets/suraj520/customer-support-ticket-dataset/customer_support_tickets.csv"
    GLOVE_PATH = "/kaggle/working/glove.6B.300d.txt"
    FOCUS      = ["Ticket Description","Ticket Subject",
                  "Ticket Priority","Ticket Type","Ticket Channel"]
    df          = pd.read_csv(DATA_PATH)[FOCUS].dropna().reset_index(drop=True)
    tokenizer   = CustomTokenizer()
    ngram_gen   = NGramGenerator((1,3))
    token_lists = tokenizer.tokenize_batch(df["Ticket Description"].tolist())
    ngram_lists = ngram_gen.generate_batch(token_lists)
    tfidf_vec   = TFIDFVectorizer(5000)
    tfidf_sparse = tfidf_vec.fit_transform(ngram_lists)
    glove        = GloVeEmbeddings(GLOVE_PATH, 300).load()
    all_vecs     = []
    for doc_idx, tokens in enumerate(token_lists):
        tfidf_scores = {}
        for token in tokens:
            if token in tfidf_vec.vocab:
                tfidf_scores[token] = tfidf_vec.idf_weights[
                    tfidf_vec.vocab[token]].item()
        all_vecs.append(glove.encode_document_tfidf_weighted(tokens, tfidf_scores))
    glove_matrix = torch.stack(all_vecs)
    tfidf_dense  = tfidf_sparse.to_dense()
    tfidf_norm   = normalize(tfidf_dense)
    glove_norm   = normalize(glove_matrix)
    return df, tfidf_norm, glove_norm, tfidf_vec, glove, tokenizer, ngram_gen

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:28px">
        🔍 HSRIS — Hybrid Semantic Retrieval & Intelligence System
    </h1>
    <p style="margin:6px 0 0; opacity:0.85; font-size:15px">
        Customer Support Ticket Search Engine | PyTorch + GloVe + TF-IDF
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────
with st.spinner("⏳ Loading models... (first run ~3 min)"):
    df, tfidf_norm, glove_norm, tfidf_vec, glove, tokenizer, ngram_gen = load_all()

st.success(f"✅ Ready | {len(df):,} tickets loaded | Device: **{device}**")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Search Settings")
    alpha = st.slider("Alpha (α)", 0.0, 1.0, 0.4, 0.05,
        help="0.0 = Pure GloVe | 1.0 = Pure TF-IDF")

    st.markdown(f"""
    | Mode | Weight |
    |------|--------|
    | TF-IDF (keyword) | `{alpha:.2f}` |
    | GloVe (semantic) | `{1-alpha:.2f}` |
    """)
    st.markdown("---")
    st.markdown("### 💡 Example Queries")
    examples = [
        "I want my money back",
        "my internet keeps dropping",
        "I was charged twice",
        "I want to cancel my subscription",
        "device stopped working after update"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

# ── Search Input ─────────────────────────────────────────────
query = st.text_area(
    "📝 Enter ticket description:",
    value=st.session_state.get("query", ""),
    height=100,
    placeholder="e.g. I want my money back for a broken product..."
)
search_clicked = st.button("🔍 Search", type="primary", use_container_width=False)

# ── Results ──────────────────────────────────────────────────
type_icons = {
    "Technical issue"      : "🔧",
    "Billing inquiry"      : "💳",
    "Refund request"       : "💰",
    "Cancellation request" : "❌",
    "Product inquiry"      : "📦"
}
priority_colors = {
    "Critical": "badge-critical",
    "High"    : "badge-high",
    "Medium"  : "badge-medium",
    "Low"     : "badge-low"
}
rank_colors = ["gold", "silver", "bronze"]

if search_clicked and query.strip():
    with st.spinner("Searching..."):
        results = search(query, alpha, df, tfidf_norm, glove_norm,
                        tfidf_vec, glove, tokenizer, ngram_gen, top_k=5)

    type_counts    = Counter([r["ticket_type"] for r in results])
    predicted_type = type_counts.most_common(1)[0][0]
    icon           = type_icons.get(predicted_type, "📋")

    st.markdown(f"""
    <div class="predicted-type">
        🎯 Predicted Ticket Type: &nbsp; {icon} {predicted_type}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Top 3 Similar Past Resolutions")

    for r in results[:3]:
        color     = rank_colors[r["rank"]-1] if r["rank"] <= 3 else "blue"
        p_class   = priority_colors.get(r["priority"], "badge-low")
        rank_medal = ["🥇","🥈","🥉"][r["rank"]-1]

        st.markdown(f"""
        <div class="result-card result-card-{color}">
            <div style="font-size:16px; font-weight:700; margin-bottom:8px">
                {rank_medal} Rank {r["rank"]} &nbsp;|&nbsp; {r["subject"]}
            </div>
            <span class="badge badge-type">{r["ticket_type"]}</span>
            <span class="badge {p_class}">{r["priority"]}</span>
            <span class="badge badge-channel">{r["channel"]}</span>
            <p style="margin:10px 0 6px; color:#ccc; font-size:14px">
                {r["description"]}...
            </p>
            <div class="score-box">
                TF-IDF: {r["tfidf_score"]:.4f} &nbsp;|&nbsp;
                GloVe: {r["glove_score"]:.4f} &nbsp;|&nbsp;
                Final: {r["final_score"]:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Full Score Breakdown")
    table_data = [{
        "Rank"        : r["rank"],
        "Subject"     : r["subject"],
        "Type"        : r["ticket_type"],
        "Priority"    : r["priority"],
        "TF-IDF"      : round(r["tfidf_score"], 4),
        "GloVe"       : round(r["glove_score"],  4),
        "Final Score" : round(r["final_score"],  4),
    } for r in results]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

elif search_clicked:
    st.warning("⚠️ Please enter a ticket description.")
