import csv
import math
import time
from collections import defaultdict, Counter
from task1 import preprocess_text
from task3 import load_passages, load_queries

def build_doc_models(passages):
    doc_texts = {}
    for qid, passages_list in passages.items():
        for pid, passage in passages_list:
            if pid not in doc_texts:
                doc_texts[pid] = passage
    doc_tf = {}
    doc_length = {}
    for pid, text in doc_texts.items():
        tokens = preprocess_text(text, remove_stopwords=True, selective_stemming=True)
        tf = Counter(tokens)
        doc_tf[pid] = tf
        doc_length[pid] = sum(tf.values())
    return doc_tf, doc_length

def build_collection_model(doc_tf, doc_length):
    collection_counter = Counter()
    for tf in doc_tf.values():
        collection_counter.update(tf)
    total_tokens = sum(doc_length.values())
    V = len(collection_counter)
    return collection_counter, total_tokens, V

def compute_query_likelihood(query_tokens, doc_tf, doc_length, collection_counter, total_tokens, V, smoothing, param):
    """
      - 'laplace': P(t|D) = (tf + 1) / (doc_length + V)
      - 'lidstone': P(t|D) = (tf + ε) / (doc_length + ε * V), with ε = param.
      - 'dirichlet': P(t|D) = (tf + µ * P(t|C)) / (doc_length + µ), with µ = param and
                     P(t|C) = (collection frequency of t) / total_tokens.
    
      log P(Q|D) = Σ_{t in Q} (qf(t) * log(P(t|D))).
    """
    log_prob = 0.0
    '''
    for token in query_tokens:
        if token in doc_tf:
            tf_d = doc_tf[token]
        else:
            tf_d = 0
    '''
    for term, qfreq in query_tokens.items():
        tf_d = doc_tf.get(term, 0)
        if smoothing == 'laplace':
            p = (tf_d + 1) / (doc_length + V)
        elif smoothing == 'lidstone':
            epsilon = param
            p = (tf_d + epsilon) / (doc_length + epsilon * V)
        elif smoothing == 'dirichlet':
            mu = param
            cf = collection_counter.get(term, 0)
            p = (tf_d + mu * (cf / total_tokens)) / (doc_length + mu)
        else:
            raise ValueError("Unknown smoothing method")
        
        if p <= 0:
            p = 1e-12
        log_prob += qfreq * math.log(p)
    return log_prob

def retrieve_query_likelihood(query, candidate_pids, doc_tf, doc_length, collection_counter, total_tokens, V, smoothing, param):
    query_tokens = preprocess_text(query, remove_stopwords=True, selective_stemming=True)
    query_counter = Counter(query_tokens)
    scores = []
    for pid in candidate_pids:
        if pid not in doc_tf:
            continue
        score = compute_query_likelihood(query_counter, doc_tf[pid], doc_length[pid],
                                         collection_counter, total_tokens, V, smoothing, param)
        scores.append((pid, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:100]

def main():
    start_time = time.perf_counter()
    
    candidate_passages = load_passages("candidate-passages-top1000.tsv")
    queries, query_order = load_queries("test-queries.tsv")

    doc_tf, doc_length = build_doc_models(candidate_passages)
    
    collection_counter, total_tokens, V = build_collection_model(doc_tf, doc_length)
    
    results_laplace = []
    results_lidstone = []
    results_dirichlet = []
    
    for qid in query_order:
        if qid not in queries:
            continue
        query_text = queries[qid]

        candidate_list = candidate_passages.get(qid, [])
        candidate_pids = [pid for pid, _ in candidate_list]
        
        laplace_candidates = retrieve_query_likelihood(query_text, candidate_pids, doc_tf, doc_length,
                                                       collection_counter, total_tokens, V,
                                                       smoothing='laplace', param=None)
        for pid, score in laplace_candidates:
            results_laplace.append((qid, pid, score))

        lidstone_candidates = retrieve_query_likelihood(query_text, candidate_pids, doc_tf, doc_length,
                                                        collection_counter, total_tokens, V,
                                                        smoothing='lidstone', param=0.1)
        for pid, score in lidstone_candidates:
            results_lidstone.append((qid, pid, score))
        
        dirichlet_candidates = retrieve_query_likelihood(query_text, candidate_pids, doc_tf, doc_length,
                                                         collection_counter, total_tokens, V,
                                                         smoothing='dirichlet', param=50)
        for pid, score in dirichlet_candidates:
            results_dirichlet.append((qid, pid, score))
    
    with open("laplace.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', lineterminator="\n")
        for qid, pid, score in results_laplace:
            writer.writerow([qid, pid, f"{score}"])
    
    with open("lidstone.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', lineterminator="\n")
        for qid, pid, score in results_lidstone:
            writer.writerow([qid, pid, f"{score}"])
    
    with open("dirichlet.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=',', lineterminator="\n")
        for qid, pid, score in results_dirichlet:
            writer.writerow([qid, pid, f"{score}"])
    
    end_time = time.perf_counter()
    print(f"Query Likelihood retrieval completed in {end_time - start_time:.4f} seconds.")
    print(f"Total rows in laplace.csv: {len(results_laplace)}")
    print(f"Total rows in lidstone.csv: {len(results_lidstone)}")
    print(f"Total rows in dirichlet.csv: {len(results_dirichlet)}")

if __name__ == "__main__":
    main()
    
'''
Query Likelihood retrieval completed in 69.1267 seconds.
Total rows in laplace.csv: 19290
Total rows in lidstone.csv: 19290
Total rows in dirichlet.csv: 19290
'''