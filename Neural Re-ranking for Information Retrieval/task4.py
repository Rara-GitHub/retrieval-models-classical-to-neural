import time
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from task2 import load_glove_embeddings, downsample_data, prepare_dataset, compute_MAP, compute_NDCG, sort_by_qid

EMBEDDING_DIM = 200  
GLOVE_FILE = "glove.6B.200d.txt"
MAX_NEGATIVES = 20
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 64

STOPWORDS = set(stopwords.words('english'))

# ---------------------- Model 1: Simple Feedforward MLP ----------------------
class ReRankingNN(nn.Module):
    def __init__(self, input_dim):
        super(ReRankingNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class RelevanceDataset(Dataset):
    def __init__(self, X, y, qids, pids):
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.float32)
        self.qids = qids
        self.pids = pids

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.qids[idx], self.pids[idx]

def train_mlp_model():
    glove_embeddings = load_glove_embeddings(GLOVE_FILE, EMBEDDING_DIM)
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype=str)
    valid_df = pd.read_csv("validation_data.tsv", sep="\t", dtype=str)
    train_df_down = downsample_data(train_df, max_negatives=MAX_NEGATIVES, random_state=42)
    
    print("Processing training data for MLP...")
    X_train, y_train, train_qids, train_pids = prepare_dataset(train_df_down, glove_embeddings, EMBEDDING_DIM)
    print("Processing validation data for MLP...")
    X_valid, y_valid, valid_qids, valid_pids = prepare_dataset(valid_df, glove_embeddings, EMBEDDING_DIM)
    
    X_train, y_train, train_qids, train_pids = sort_by_qid(X_train, y_train, train_qids, train_pids)
    X_valid, y_valid, valid_qids, valid_pids = sort_by_qid(X_valid, y_valid, valid_qids, valid_pids)
    
    train_dataset = RelevanceDataset(X_train, y_train, train_qids, train_pids)
    valid_dataset = RelevanceDataset(X_valid, y_valid, valid_qids, valid_pids)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_dim = 2 * EMBEDDING_DIM + 1
    model = ReRankingNN(input_dim)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    mlp_train_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for features, labels, _, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)
        epoch_loss /= len(train_loader.dataset)
        mlp_train_losses.append(epoch_loss)
        print(f"MLP Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
    
    model.eval()
    mlp_preds = []
    mlp_ids = []
    with torch.no_grad():
        for features, labels, qids, pids in valid_loader:
            outputs = model(features)
            mlp_preds.extend(outputs.squeeze().cpu().numpy())
            mlp_ids.extend(list(zip(qids, pids)))
            
    mlp_predictions = {}
    mlp_ground_truth = {}
    for i, (qid, pid) in enumerate(mlp_ids):
        mlp_predictions.setdefault(qid, {})[pid] = mlp_preds[i]
        mlp_ground_truth.setdefault(qid, {})[pid] = y_valid[i][0]
        
    mlp_map = compute_MAP(mlp_predictions, mlp_ground_truth)
    mlp_ndcg = compute_NDCG(mlp_predictions, mlp_ground_truth)
    print(f"\nMLP Validation mAP: {mlp_map:.4f}, NDCG: {mlp_ndcg:.4f}")
    
    return mlp_map, mlp_ndcg, mlp_train_losses

# ---------------------- Model 2: Siamese Re-Ranking Network ----------------------
def clean_and_tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return tokens

def text_to_sequence(text, embeddings, max_len, emb_dim=EMBEDDING_DIM):
    tokens = clean_and_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    seq = []
    for t in tokens:
        if t in embeddings:
            seq.append(embeddings[t])
        else:
            seq.append(np.zeros(emb_dim))
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        pad_len = max_len - len(seq)
        seq.extend([np.zeros(emb_dim)] * pad_len)
    return np.array(seq)

class SharedBiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1, dropout=0.3):
        super(SharedBiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
    
    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat((h_forward, h_backward), dim=1)

class SiameseReRankingNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SiameseReRankingNN, self).__init__()
        self.encoder = SharedBiLSTMEncoder(embedding_dim, hidden_dim)
        combined_dim = 8 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, q_seq, q_len, p_seq, p_len):
        q_repr = self.encoder(q_seq, q_len)
        p_repr = self.encoder(p_seq, p_len)
        diff = torch.abs(q_repr - p_repr)
        prod = q_repr * p_repr
        combined = torch.cat([q_repr, p_repr, diff, prod], dim=1)
        return self.classifier(combined)

class RelevanceSequenceDataset(Dataset):
    def __init__(self, df, embeddings, query_max_len, passage_max_len):
        self.queries = df['queries'].tolist()
        self.passages = df['passage'].tolist()
        self.labels = df['relevancy'].astype(float).tolist()
        self.qids = df['qid'].tolist()
        self.pids = df['pid'].tolist()
        self.embeddings = embeddings
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        q_text = self.queries[idx]
        p_text = self.passages[idx]
        q_seq = text_to_sequence(q_text, self.embeddings, self.query_max_len)
        p_seq = text_to_sequence(p_text, self.embeddings, self.passage_max_len)
        q_len = min(len(clean_and_tokenize(q_text)), self.query_max_len)
        p_len = min(len(clean_and_tokenize(p_text)), self.passage_max_len)
        return (torch.tensor(q_seq, dtype=torch.float32),
                torch.tensor([q_len], dtype=torch.int64),
                torch.tensor(p_seq, dtype=torch.float32),
                torch.tensor([p_len], dtype=torch.int64),
                torch.tensor(self.labels[idx], dtype=torch.float32),
                self.qids[idx],
                self.pids[idx])

def collate_fn(batch):
    q_seqs = torch.stack([item[0] for item in batch])
    q_lens = torch.cat([item[1] for item in batch])
    p_seqs = torch.stack([item[2] for item in batch])
    p_lens = torch.cat([item[3] for item in batch])
    labels = torch.stack([item[4] for item in batch])
    qids = [item[5] for item in batch]
    pids = [item[6] for item in batch]
    return q_seqs, q_lens, p_seqs, p_lens, labels, qids, pids

def train_siamese_model():
    QUERY_MAX_LEN = 20
    PASSAGE_MAX_LEN = 100
    HIDDEN_DIM = 128
    
    glove_embeddings = load_glove_embeddings(GLOVE_FILE, EMBEDDING_DIM)
    
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype=str)
    valid_df = pd.read_csv("validation_data.tsv", sep="\t", dtype=str)
    
    train_df_down = downsample_data(train_df, max_negatives=MAX_NEGATIVES, random_state=42)
    train_df_down = train_df_down.sort_values(by='qid')
    valid_df = valid_df.sort_values(by='qid')
    
    train_dataset = RelevanceSequenceDataset(train_df_down, glove_embeddings, QUERY_MAX_LEN, PASSAGE_MAX_LEN)
    valid_dataset = RelevanceSequenceDataset(valid_df, glove_embeddings, QUERY_MAX_LEN, PASSAGE_MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = SiameseReRankingNN(EMBEDDING_DIM, HIDDEN_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    siamese_train_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for q_seqs, q_lens, p_seqs, p_lens, labels, _, _ in train_loader:
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(q_seqs, q_lens, p_seqs, p_lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * q_seqs.size(0)
        epoch_loss /= len(train_loader.dataset)
        siamese_train_losses.append(epoch_loss)
        print(f"Siamese Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.6f}")
    
    model.eval()
    siamese_preds = []
    siamese_ids = []
    with torch.no_grad():
        for q_seqs, q_lens, p_seqs, p_lens, labels, qids, pids in valid_loader:
            outputs = model(q_seqs, q_lens, p_seqs, p_lens)
            siamese_preds.extend(outputs.squeeze().cpu().numpy())
            siamese_ids.extend(list(zip(qids, pids)))
    
    siamese_predictions = {}
    siamese_ground_truth = {}
    for i, (qid, pid) in enumerate(siamese_ids):
        siamese_predictions.setdefault(qid, {})[pid] = siamese_preds[i]
        siamese_ground_truth.setdefault(qid, {})[pid] = valid_dataset.labels[i]
        
    siamese_map = compute_MAP(siamese_predictions, siamese_ground_truth)
    siamese_ndcg = compute_NDCG(siamese_predictions, siamese_ground_truth)
    print(f"\nSiamese Validation mAP: {siamese_map:.4f}, NDCG: {siamese_ndcg:.4f}")
    
    return siamese_map, siamese_ndcg, siamese_train_losses, model

def main():
    start_time = time.time()
    global glove_embeddings
    glove_embeddings = load_glove_embeddings(GLOVE_FILE, EMBEDDING_DIM)
    
    print("Training Simple MLP Model...")
    mlp_map, mlp_ndcg, mlp_losses = train_mlp_model()
    
    print("\nTraining Siamese Re-ranking Model...")
    siamese_map, siamese_ndcg, siamese_losses, siamese_model = train_siamese_model()
    torch.save(siamese_model.state_dict(), "siamese_model.pt")
    
    print("\nComparison of Models:")
    print(f"Simple MLP - mAP: {mlp_map:.4f}, NDCG: {mlp_ndcg:.4f}")
    print(f"Siamese   - mAP: {siamese_map:.4f}, NDCG: {siamese_ndcg:.4f}")
    
    QUERY_MAX_LEN = 20
    PASSAGE_MAX_LEN = 100
    
    print("Processing test data for NN...")

    test_queries = pd.read_csv("test-queries.tsv", sep="\t", header=None,
                                names=["qid", "queries"], dtype=str)
    test_candidates = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", header=None,
                                names=["qid", "pid", "query", "passage"], dtype=str)

    filtered_candidates = test_candidates[test_candidates["qid"].isin(test_queries["qid"])]
    filtered_candidates.rename(columns={"query": "queries"}, inplace=True)

    if "relevancy" not in filtered_candidates.columns:
        filtered_candidates["relevancy"] = "0.0"

    test_dataset = RelevanceSequenceDataset(filtered_candidates, glove_embeddings, QUERY_MAX_LEN, PASSAGE_MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    siamese_model.eval()
    test_predictions = {}
    with torch.no_grad():
        for q_seqs, q_lens, p_seqs, p_lens, labels, qids, pids in test_loader:
            outputs = siamese_model(q_seqs, q_lens, p_seqs, p_lens)
            preds = outputs.squeeze().cpu().numpy()
            for i, qid in enumerate(qids):
                pid = pids[i]
                score = preds[i]
                test_predictions.setdefault(qid, {})[pid] = score

    with open("NN.txt", "w") as f:
        for qid, scores in test_predictions.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_scores, start=1):
                if rank > 100:
                    break
                f.write(f"{qid} A2 {pid} {rank} {score:.4f} NN\n")
                
    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time)/60:.2f} minutes")
    
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, len(mlp_losses)+1), mlp_losses, marker='o', label="Simple MLP")
    plt.plot(range(1, len(siamese_losses)+1), siamese_losses, marker='o', label="Siamese")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("model_comparison_loss.png")
    plt.show()

if __name__ == "__main__":
    main()