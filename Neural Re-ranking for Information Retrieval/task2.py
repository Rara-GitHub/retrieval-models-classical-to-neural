import os
import re
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

EMBEDDING_DIM = 200  
GLOVE_FILE = "glove.6B.200d.txt"

LEARNING_RATES = [1e-3, 1e-2, 1e-1,0.5,1]
EPOCHS = 300
BATCH_SIZE = 64
TOLERANCE = 1e-6

STOPWORDS = set(stopwords.words('english'))

def load_glove_embeddings(glove_file, emb_dim=EMBEDDING_DIM):
    """
    Loads pre-trained GloVe embeddings from a file.
    Returns a dictionary mapping words to embedding vectors.
    """
    embeddings = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) == emb_dim:
                embeddings[word] = vec
    return embeddings

if not os.path.exists(GLOVE_FILE):
    raise FileNotFoundError(f"GloVe embeddings not found at {GLOVE_FILE}.")

glove_embeddings = load_glove_embeddings(GLOVE_FILE, EMBEDDING_DIM)

def clean_and_tokenize(text):
    """
    Lowercases, removes non-alphabetical characters, and tokenizes the text.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return tokens

def average_embedding(text, embeddings, emb_dim=EMBEDDING_DIM):
    """
    Computes the average embedding vector for a given text using the provided embeddings.
    If no known words are found, returns a zero vector.
    """
    tokens = clean_and_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    vectors = [embeddings[t] for t in tokens if t in embeddings]
    if vectors:
        return np.mean(np.vstack(vectors), axis=0)
    else:
        return np.zeros(emb_dim)

def create_feature(query, passage, embeddings, emb_dim=EMBEDDING_DIM, add_bias=True):
    """
    Creates a feature vector by concatenating the average query and passage embeddings.
    Optionally adds a bias term at the beginning.
    """
    query_vec = average_embedding(query, embeddings, emb_dim)
    passage_vec = average_embedding(passage, embeddings, emb_dim)
    feature = np.concatenate([query_vec, passage_vec])
    if add_bias:
        feature = np.concatenate(([1.0], feature))
    return feature

def downsample_data(df, max_negatives=20, random_state=42):
    """
    For each query (qid) in the DataFrame, retains all positive records and samples a
    maximum of `max_negatives` negatives. Returns the downsampled DataFrame.
    """
    np.random.seed(random_state)
    sampled_groups = []
    for qid in df['qid'].unique():
        group = df[df['qid'] == qid]
        pos = group[group['relevancy'] == "1.0"]
        neg = group[group['relevancy'] != "1.0"]
        # Allowed negatives is max_negatives minus number of positives
        n_neg = max(0, max_negatives - len(pos))
        if len(neg) > n_neg:
            neg = neg.sample(n=n_neg, random_state=random_state)
        sampled_groups.append(pd.concat([pos, neg]))
    downsampled_df = pd.concat(sampled_groups).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return downsampled_df


def prepare_dataset(df, embeddings, emb_dim=EMBEDDING_DIM):
    """
    Converts the DataFrame's query-passage pairs to features (concatenated embeddings)
    and the relevance labels.
    Returns: X (np.array), y (np.array), and lists of query and passage IDs.
    """
    features, labels, qids, pids = [], [], [], []
    for _, row in df.iterrows():
        feat = create_feature(row['queries'], row['passage'], embeddings, emb_dim, add_bias=True)
        features.append(feat)
        labels.append(float(row['relevancy']))
        qids.append(row['qid'])
        pids.append(row['pid'])
    X = np.vstack(features)
    y = np.array(labels).reshape(-1, 1)
    return X, y, qids, pids

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred):
    """
    Computes the binary cross-entropy loss.
    Adds a small epsilon to avoid log(0).
    """
    eps = 1e-10
    return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

def compute_gradient(X, y, y_pred):
    """
    Computes the gradient of the loss with respect to theta.
    """
    m = X.shape[0]
    return (X.T @ (y_pred - y)) / m

def train_logistic_regression(X, y, lr=1e-3, epochs=500, batch_size=64, tol=1e-6, verbose=False):
    """
    Trains a logistic regression model using mini-batch gradient descent.
    Returns the learned parameters theta and a list of epoch losses.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    losses = []
    n_batches = int(np.ceil(m / batch_size))
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        epoch_loss = 0.0
        
        for start in range(0, m, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            z = X_batch @ theta
            y_pred = sigmoid(z)
            loss = compute_loss(y_batch, y_pred)
            epoch_loss += loss
            grad = compute_gradient(X_batch, y_batch, y_pred)
            theta -= lr * grad
        epoch_loss /= n_batches
        losses.append(epoch_loss)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        if epoch > 0 and abs(losses[-2] - epoch_loss) / losses[-2] < tol:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}.")
            break
    return theta, losses

def compute_MAP(pred_dict, ground_truth):
    total_ap = 0.0
    query_count = 0
    for qid, scores in pred_dict.items():
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        num_relevant = 0
        ap = 0.0
        for rank, (pid, score) in enumerate(sorted_scores, start=1):
            if ground_truth.get(qid, {}).get(pid, 0) == 1.0:
                num_relevant += 1
                ap += num_relevant / rank
        if num_relevant > 0:
            total_ap += ap / num_relevant
            query_count += 1
    return total_ap / query_count if query_count > 0 else 0.0

def compute_NDCG(pred_dict, ground_truth):
    total_ndcg = 0.0
    query_count = 0
    for qid, scores in pred_dict.items():
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        dcg = 0.0
        for rank, (pid, score) in enumerate(sorted_scores, start=1):
            if ground_truth.get(qid, {}).get(pid, 0) == 1.0:
                dcg += 1 / math.log2(rank + 1)
        # Compute ideal DCG
        ideal_rels = sum(1 for pid in ground_truth.get(qid, {}) if ground_truth[qid][pid] == 1.0)
        ideal_dcg = sum(1 / math.log2(i + 2) for i in range(ideal_rels))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        total_ndcg += ndcg
        query_count += 1
    return total_ndcg / query_count if query_count > 0 else 0.0


def sort_by_qid(X, y, qids, pids):
    df = pd.DataFrame({'qid': qids, 'pid': pids})
    df['index'] = np.arange(len(qids))
    df_sorted = df.sort_values(by='qid')
    indices = df_sorted['index'].values
    X_sorted = X[indices]
    y_sorted = y[indices]
    qids_sorted = df_sorted['qid'].tolist()
    pids_sorted = df_sorted['pid'].tolist()
    return X_sorted, y_sorted, qids_sorted, pids_sorted
def main():
    start_time = time.time()
    
    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype=str)
    valid_df = pd.read_csv("validation_data.tsv", sep="\t", dtype=str)
    
    train_df_down = downsample_data(train_df, max_negatives=20, random_state=42)
    
    print("Processing training data...")
    X_train, y_train, train_qids, train_pids = prepare_dataset(train_df_down, glove_embeddings, EMBEDDING_DIM)
    print("Processing validation data...")
    X_valid, y_valid, valid_qids, valid_pids = prepare_dataset(valid_df, glove_embeddings, EMBEDDING_DIM)
    
    loss_histories = {}
    models = {}
    
    for lr in LEARNING_RATES:
        print(f"\nTraining with learning rate: {lr}")
        theta, losses = train_logistic_regression(X_train, y_train, lr=lr, epochs=EPOCHS,
                                                  batch_size=BATCH_SIZE, tol=TOLERANCE, verbose=True)
        models[lr] = theta
        loss_histories[lr] = losses

    def predict(X, theta):
        return sigmoid(X @ theta)
    
    map_scores = {}
    ndcg_scores = {}
    for lr, theta in models.items():
        y_pred_valid = predict(X_valid, theta)
    
        valid_predictions = {}
        valid_ground_truth = {}
        for i, qid in enumerate(valid_qids):
            pid = valid_pids[i]
            score = y_pred_valid[i][0]
            valid_predictions.setdefault(qid, {})[pid] = score
            valid_ground_truth.setdefault(qid, {})[pid] = y_valid[i][0]
    
        map_score = compute_MAP(valid_predictions, valid_ground_truth)
        ndcg_score = compute_NDCG(valid_predictions, valid_ground_truth)
        map_scores[lr] = map_score
        ndcg_scores[lr] = ndcg_score
        print(f"Learning Rate: {lr} --> Validation MAP: {map_score:.4f}, NDCG: {ndcg_score:.4f}")
     
    best_lr = max(map_scores, key=map_scores.get)
    print(f"\nBest learning rate by MAP: {best_lr}")
    best_theta = models[best_lr]

    print("Processing test data for LR...")

    test_queries = pd.read_csv("test-queries.tsv", sep="\t", header=None,
                                 names=["qid", "queries"], dtype=str)
    test_candidates = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", header=None,
                                  names=["qid", "pid", "query", "passage"], dtype=str)
    
    filtered_candidates = test_candidates[test_candidates["qid"].isin(test_queries["qid"])]
    filtered_candidates.rename(columns={"query": "queries"}, inplace=True)

    if "relevancy" not in filtered_candidates.columns:
        filtered_candidates["relevancy"] = "0.0"
    
    X_test, _, test_qids, test_pids = prepare_dataset(filtered_candidates, glove_embeddings, EMBEDDING_DIM)
    
    y_pred_test = predict(X_test, best_theta)
    
    test_predictions = {}
    for i, qid in enumerate(test_qids):
        pid = test_pids[i]
        score = y_pred_test[i][0]
        test_predictions.setdefault(qid, {})[pid] = score

    with open("LR.txt", "w") as f:
        for qid, scores in test_predictions.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_scores, start=1):
                if rank > 100:
                    break
                f.write(f"{qid} A2 {pid} {rank} {score:.4f} LR\n")
    
    plt.figure(figsize=(8, 6))
    for lr in LEARNING_RATES:
        plt.plot(range(1, len(loss_histories[lr]) + 1), loss_histories[lr], label=f"LR = {lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch for Different Learning Rates")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curves.png")
    plt.show()

    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time)/60:.2f} minutes")
    
if __name__ == "__main__":
    main()