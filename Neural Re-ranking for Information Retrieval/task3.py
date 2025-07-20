import os
import time
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from xgboost import XGBRanker

from task2 import (
    load_glove_embeddings, 
    downsample_data, 
    compute_MAP, 
    compute_NDCG, 
    sort_by_qid
)

EMBEDDING_DIM = 200
GLOVE_FILE = "glove.6B.200d.txt"
MAX_NEGATIVES = 20

STOPWORDS = set(stopwords.words('english'))
if not os.path.exists(GLOVE_FILE):
    raise FileNotFoundError(f"GloVe embeddings not found at {GLOVE_FILE}.")
glove_embeddings = load_glove_embeddings(GLOVE_FILE, EMBEDDING_DIM)

def prepare_dataset_with_features(df, glove_embeddings, embedding_dim):
    """
    Prepares a dataset with two sets of features:
      • Glove embedding features: Compute average embeddings for query and passage,
        along with their element-wise difference.
      • Four additional features:
         1. Query Length: Number of non-stopword tokens in the query.
         2. Passage Length: Number of non-stopword tokens in the passage.
         3. Word Overlap Count: Count of common tokens between query and passage.
         4. Jaccard Similarity: Intersection-over-union of token sets from query and passage.
    """
    X, y, qids, pids = [], [], [], []
    
    for index, row in df.iterrows():
        query = row['queries']
        passage = row['passage']
        label = float(row['relevancy'])
        qid = row['qid']
        pid = row['pid']
        
        query_tokens = [token for token in word_tokenize(query.lower()) if token not in STOPWORDS]
        passage_tokens = [token for token in word_tokenize(passage.lower()) if token not in STOPWORDS]
        
        query_emb = np.mean([glove_embeddings.get(token, np.zeros(embedding_dim)) for token in query_tokens], axis=0)
        passage_emb = np.mean([glove_embeddings.get(token, np.zeros(embedding_dim)) for token in passage_tokens], axis=0)
        
        emb_diff = query_emb - passage_emb
        emb_features = np.concatenate([query_emb, passage_emb, emb_diff])
        
        # Feature 1: Query Length (number of tokens)
        query_length = len(query_tokens)
        # Feature 2: Passage Length (number of tokens)
        passage_length = len(passage_tokens)
        # Feature 3: Word Overlap Count
        overlap_count = len(set(query_tokens).intersection(set(passage_tokens)))
        # Feature 4: Jaccard Similarity
        union_count = len(set(query_tokens).union(set(passage_tokens)))
        jaccard_similarity = overlap_count / union_count if union_count > 0 else 0
        
        extra_features = np.array([query_length, passage_length, overlap_count, jaccard_similarity])
        
        features = np.concatenate([emb_features, extra_features])
        
        X.append(features)
        y.append([label])
        qids.append(qid)
        pids.append(pid)
        
    return np.array(X), np.array(y), qids, pids

def create_group_array(qids):
    """Creates an array indicating the number of documents per query (assumes sorted qids)."""
    groups = []
    current_qid = qids[0]
    count = 0
    for q in qids:
        if q == current_qid:
            count += 1
        else:
            groups.append(count)
            current_qid = q
            count = 1
    groups.append(count)
    return groups

def main():
    start_time = time.time()

    train_df = pd.read_csv("train_data.tsv", sep="\t", dtype=str)
    valid_df = pd.read_csv("validation_data.tsv", sep="\t", dtype=str)
    
    train_df_down = downsample_data(train_df, max_negatives=MAX_NEGATIVES, random_state=42)
    
    print("Processing training data...")
    X_train, y_train, train_qids, train_pids = prepare_dataset_with_features(train_df_down, glove_embeddings, EMBEDDING_DIM)
    print("Processing validation data...")
    X_valid, y_valid, valid_qids, valid_pids = prepare_dataset_with_features(valid_df, glove_embeddings, EMBEDDING_DIM)
    
    X_train, y_train, train_qids, train_pids = sort_by_qid(X_train, y_train, train_qids, train_pids)
    X_valid, y_valid, valid_qids, valid_pids = sort_by_qid(X_valid, y_valid, valid_qids, valid_pids)
    
    group_train = create_group_array(train_qids)
    group_valid = create_group_array(valid_qids)
    
    # Hyper-parameter tuning setup
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 300, 500],
        'objective': ['rank:pairwise']
    }
    
    best_map = -1
    best_ndcg = -1
    best_params = None
    best_model = None
    results = []
    
    # Grid search over hyper-parameters
    for lr in param_grid['learning_rate']:
        for md in param_grid['max_depth']:
            for n_est in param_grid['n_estimators']:
                print(f"\nTraining XGBRanker with learning_rate={lr}, max_depth={md}, n_estimators={n_est}")
                model = XGBRanker(learning_rate=lr,
                                  max_depth=md,
                                  n_estimators=n_est,
                                  objective='rank:pairwise',
                                  verbosity=0)
                model.fit(X_train, y_train, group=group_train,
                          eval_set=[(X_valid, y_valid)], eval_group=[group_valid],
                          verbose=False)
                
                # Predict and reorganize predictions by query
                y_pred = model.predict(X_valid)
                valid_predictions = {}
                valid_ground_truth = {}
                for i, qid in enumerate(valid_qids):
                    pid = valid_pids[i]
                    score = y_pred[i]
                    valid_predictions.setdefault(qid, {})[pid] = score
                    valid_ground_truth.setdefault(qid, {})[pid] = y_valid[i][0]
                
                current_map = compute_MAP(valid_predictions, valid_ground_truth)
                current_ndcg = compute_NDCG(valid_predictions, valid_ground_truth)
                print(f"mAP: {current_map:.4f}, NDCG: {current_ndcg:.4f}")
                
                results.append({
                    'learning_rate': lr,
                    'max_depth': md,
                    'n_estimators': n_est,
                    'mAP': current_map,
                    'NDCG': current_ndcg
                })
                
                # Keep track of the best performing model (based on mAP)
                if current_map > best_map:
                    best_map = current_map
                    best_ndcg = current_ndcg
                    best_params = {'learning_rate': lr, 'max_depth': md, 'n_estimators': n_est}
                    best_model = model

    print("\nHyper-parameter tuning results:")
    for res in results:
        print(res)
    print("\nBest hyper-parameters:", best_params)
    print(f"Best validation mAP: {best_map:.4f}, NDCG: {best_ndcg:.4f}")
    
    
    print("\nProcessing test data for LM...")

    test_queries = pd.read_csv("test-queries.tsv", sep="\t", header=None,
                                 names=["qid", "queries"], dtype=str)
    test_candidates = pd.read_csv("candidate_passages_top1000.tsv", sep="\t", header=None,
                                  names=["qid", "pid", "query", "passage"], dtype=str)
    
    filtered_candidates = test_candidates[test_candidates["qid"].isin(test_queries["qid"])]
    filtered_candidates.rename(columns={"query": "queries"}, inplace=True)

    if "relevancy" not in filtered_candidates.columns:
        filtered_candidates["relevancy"] = "0.0"
    
    X_test, _, test_qids, test_pids = prepare_dataset_with_features(filtered_candidates, glove_embeddings, EMBEDDING_DIM)
    
    y_pred_test = best_model.predict(X_test)
    
    test_predictions = {}
    for i, qid in enumerate(test_qids):
        pid = test_pids[i]
        score = y_pred_test[i]
        test_predictions.setdefault(qid, {})[pid] = score

    with open("LM.txt", "w") as f:
        for qid, scores in test_predictions.items():
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (pid, score) in enumerate(sorted_scores, start=1):
                if rank > 100:
                    break
                f.write(f"{qid} A2 {pid} {rank} {score:.4f} LM\n")

    
    end_time = time.time()
    print(f"\nTotal processing time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()