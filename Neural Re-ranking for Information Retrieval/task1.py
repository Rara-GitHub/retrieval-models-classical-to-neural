import re
import math
import nltk
import pandas as pd
from timeit import default_timer as timer
import ssl

nltk.download('stopwords')
nltk.download('punkt')


class BM25Tokeniser:
    def __init__(self):
        self.stemmer = nltk.stem.SnowballStemmer("english")
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
    
    def tokenize(self, text):
        """
        Tokenize the input text by:
         - Converting to lowercase.
         - Replacing non-alphanumeric characters with spaces.
         - Splitting by whitespace.
         - Removing stopwords.
         - Applying stemming.
        Returns a list of processed tokens.
        """
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stopwords]
        return tokens

tokeniser = BM25Tokeniser()

############################################
# 1. Preprocessing: Process Passages and Queries
############################################

def process_passages(passages_series):
    """
    Processes a pandas Series of passages using BM25Tokeniser.
    Returns a list of token lists.
    """
    return [tokeniser.tokenize(text) for text in passages_series]

def process_queries(queries_series):
    """
    Processes a pandas Series of queries using BM25Tokeniser.
    Returns a list of token lists.
    """
    return [tokeniser.tokenize(text) for text in queries_series]

############################################
# 2. Build Inverted Index
############################################

def build_inverted_index(passages):
    """
    Given a dictionary mapping passage IDs to token lists,
    constructs an inverted index mapping each token to a dictionary
    of {passage_id: frequency}.
    """
    index = {}
    for pid, tokens in passages.items():
        # Use a set to process each unique token per passage.
        for token in set(tokens):
            freq = tokens.count(token)
            if token not in index:
                index[token] = {pid: freq}
            else:
                index[token][pid] = freq
    return index

############################################
# 3. BM25 Scoring Function
############################################

def bm25_scoring(qid_pid_df, inv_index, query_dict, passage_dict, k1=1.2, k2=100, b=0.75):
    """
    Computes BM25 scores for each (query, passage) pair provided in qid_pid_df.
    Returns a DataFrame with columns: qid, pid, and bm25_score.
    """
    results = pd.DataFrame()
    total_docs = len(passage_dict)
    avg_dl = sum(len(tokens) for tokens in passage_dict.values()) / total_docs

    for qid in qid_pid_df['qid'].unique():
        q_tokens = query_dict.get(qid)
        candidate_ids = qid_pid_df[qid_pid_df['qid'] == qid]['pid'].tolist()
        scores = []
        for pid in candidate_ids:
            score = 0.0
            p_tokens = passage_dict.get(pid)
            dl = len(p_tokens)
            norm = k1 * ((1 - b) + b * (dl / avg_dl))
            common_tokens = set(q_tokens) & set(p_tokens)
            for token in common_tokens:
                posting = inv_index.get(token, {})
                doc_freq = len(posting)
                term_freq = posting.get(pid, 0)
                query_freq = q_tokens.count(token)
                if term_freq > 0 and query_freq > 0:
                    idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                    score += idf * ((k1 + 1) * term_freq / (norm + term_freq)) * ((k2 + 1) * query_freq / (k2 + query_freq))
            scores.append((qid, pid, score))
        df_query = pd.DataFrame(scores, columns=['qid', 'pid', 'score'])
        df_query = df_query.sort_values(by='score', ascending=False)
        results = pd.concat([results, df_query], ignore_index=True)
    results.columns = ['qid', 'pid', 'bm25_score']
    return results

############################################
# 4. Evaluation Metrics: MAP and NDCG
############################################

def calculate_MAP(predictions, ground_truth):
    """
    Computes Mean Average Precision (MAP) for predictions.
    Only considers passages with relevancy "1.0" as relevant.
    """
    total_ap = 0.0
    for qid, rank_dict in predictions.items():
        ap = 0.0
        rel_count = 0
        for rank, pid in enumerate(rank_dict.keys(), start=1):
            if pid in ground_truth.get(qid, {}) and ground_truth[qid][pid] == "1.0":
                rel_count += 1
                ap += rel_count / rank
        denom = min(len(ground_truth.get(qid, {})), len(rank_dict))
        if denom > 0:
            ap /= denom
        total_ap += ap
    return total_ap / len(predictions)

def calculate_NDCG(predictions, ground_truth):
    """
    Computes Mean NDCG for the predictions.
    Only passages with relevancy "1.0" are considered relevant.
    """
    total_ndcg = 0.0
    for qid, rank_dict in predictions.items():
        dcg = 0.0
        for rank, pid in enumerate(rank_dict.keys(), start=1):
            if pid in ground_truth.get(qid, {}) and ground_truth[qid][pid] == "1.0":
                dcg += 1 / math.log2(rank + 1)
        num_relevants = sum(1 for pid in ground_truth.get(qid, {}) if ground_truth[qid][pid] == "1.0")
        idcg = sum(1 / math.log2(i + 2) for i in range(num_relevants))
        total_ndcg += (dcg / idcg) if idcg > 0 else 0.0
    return total_ndcg / len(predictions)

############################################
# 5. Main Evaluation Pipeline
############################################

def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    start_time = timer()
    
    df_val = pd.read_csv("validation_data.tsv", sep="\t", dtype="string")

    tokenised_passages = process_passages(df_val['passage'])
    passages_dict = dict(zip(df_val['pid'].tolist(), tokenised_passages))
    
    inv_index = build_inverted_index(passages_dict)
    
    tokenised_queries = process_queries(df_val['queries'])
    queries_dict = dict(zip(df_val['qid'].tolist(), tokenised_queries))
    
    df_qid_pid = df_val[['qid', 'pid']]
    
    df_bm25 = bm25_scoring(df_qid_pid, inv_index, queries_dict, passages_dict)
    
    bm25_grouped = (
        df_bm25.groupby('qid')
                .apply(lambda g: dict(zip(g['pid'], g['bm25_score'])), include_groups=False)
                .to_dict()
    )
    
    gt_relevance = (
        df_val[df_val['relevancy'] == "1.0"]
        .groupby('qid')
        .apply(lambda g: dict(zip(g['pid'], g['relevancy'])), include_groups=False)
        .to_dict()
    )
    
    map_score = calculate_MAP(bm25_grouped, gt_relevance)
    ndcg_score = calculate_NDCG(bm25_grouped, gt_relevance)
    
    end_time = timer()
    elapsed_minutes = (end_time - start_time) / 60
    
    print("BM25 MAP:", map_score)
    print("BM25 NDCG:", ndcg_score)
    print(f"Processing time: {elapsed_minutes:.1f} minutes")

if __name__ == "__main__":
    main()