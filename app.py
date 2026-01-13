import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from collections import defaultdict
from typing import List
from multiprocessing import Pool, cpu_count

# --- BM25 Implementation ---
class BM25:
    def __init__(self, corpus: List[str], tokenizer):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        # word -> number of documents with word
        nd = defaultdict(int)
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = defaultdict(int)  # word: frequency
            for word in document:
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, _ in frequencies.items():
                nd[word] += 1
            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query: str, documents, n=5):
        assert self.corpus_size == len(
            documents
        ), "The documents given don't match the index corpus!"
        if self.tokenizer:
            query = self.tokenizer(query)
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n], scores[top_n]

    def eval_metr(self, score, l_score, n=10):
        # normalise scores
        norm_score = normalize_array(score)
        norm_l_score = normalize_array(l_score)
        """
        #rev sort top_n l_score and store doc_id/index
        rev_norm_l_score = sorted(norm_l_score, reverse = True)
        top_n = n
        idx_list = []
        for i in range(top_n):
            idx_list.append(norm_l_score.get(rev_norm_l_score[i]))
        """
        # calc scores
        # relevance scoring can be 0,1,2,3; so we take 0.98 and above as reasonable for tp/fp ~98%
        ml_norm_score = [int(i > 0.9995) for i in norm_score]
        ml_norm_l_score = [int(i > 0.9995) for i in norm_l_score]

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(n):
            if ml_norm_l_score[i] == 1 and ml_norm_score[i] == 1:
                tp += 1
            if ml_norm_l_score[i] == 0 and ml_norm_score[i] == 0:
                tn += 1
            if ml_norm_l_score[i] == 1 and ml_norm_score[i] == 0:
                fp += 1
            if ml_norm_l_score[i] == 0 and ml_norm_score[i] == 1:
                fn += 1
        # metric set 1
        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        # accuracy = tp/(tp+fp+tn+fn)
        if (2 * tp + fp + fn) != 0:
            f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            f1 = 0
        # metric set 2

        ndcg = 0
        # ndcg
        dcg = l_score[0]
        for i in range(1, len(l_score)):
            dcg += l_score[i] / (np.log2(i + 1))
        idcg = score[0]
        for i in range(1, len(score)):
            idcg += score[i] / (np.log2(i + 1))

        # normalized discounted cumulative gain (NDCG) calculation
        if idcg == 0:
            ndcg = 0
        else:
            ndcg = dcg / idcg

        return [precision, recall, f1, ndcg]

class BM25Okapi:
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer
        if tokenizer:
            corpus = self._tokenize_corpus(corpus)
        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _tokenize_corpus(self, corpus):
        return [self.tokenizer(doc) for doc in corpus]

    def _initialize(self, corpus):
        nd = defaultdict(int)
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)
            frequencies = defaultdict(int)
            for word in document:
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] += 1
            self.corpus_size += 1
        self.avgdl = num_doc / self.corpus_size
        return nd

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_top_n(self, query, documents, n=20):
        assert self.corpus_size == len(documents), "Documents don't match corpus size!"
        if self.tokenizer:
            query = self.tokenizer(query)
        scores = self.get_scores(query)
        top_n_indices = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n_indices], scores[top_n_indices]

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score.tolist()

    def get_bm25scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log((self.corpus_size + 1) / freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()

    def get_bm25scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

# --- Load Dataset ---
def load_data():
    df = pd.read_csv("politik_merge.csv")
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Judul", "Content"], inplace=True)
    return df

# --- Simple Tokenizer ---
def simple_tokenizer(text):
    return re.findall(r"\w+", text.lower())

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Pencarian Berita Politik", layout="wide")

    # --- CSS ---
    st.markdown("""
        <style>
        body {
            background-color: #f4f4f4;
        }
        .result-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .title {
            color: #1f77b4;
            font-size: 24px;
            font-weight: bold;
        }
        .info {
            color: #333;
            font-size: 14px;
            margin-top: 5px;
        }
        .score {
            color: green;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Judul ---
    st.title("📚 Pencarian Berita Politik dengan BM25")

    df = load_data()
    total_artikel = len(df)
    total_sumber = df["source"].nunique()

    st.markdown(
        f"<div style='background:#dceefb;padding:15px;border-radius:10px;margin-bottom:10px;font-size:18px;'>"
        f"📝 <b>{total_artikel:,}</b> artikel dari <b>{total_sumber}</b> sumber berita siap untuk dicari!"
        f"</div>", unsafe_allow_html=True
    )

    # --- Pilihan Varian BM25 ---
    bm25_variants = {
        "BM25 Okapi": BM25Okapi,
        "BM25L": BM25L,
        "BM25Plus": BM25Plus
    }
    
    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        query = st.text_input("🔎 Masukkan Kata Kunci:")
    with col2:
        selected_source = st.selectbox("📰 Pilih Sumber Berita:", ["Semua"] + sorted(set(df["source"].dropna().astype(str).unique())))
    with col3:
        jumlah_dokumen = st.number_input("📄 Jumlah Hasil Ditampilkan:", min_value=1, max_value=100, value=10)

    selected_variant_name = st.selectbox("⚙️ Pilih Varian BM25:", list(bm25_variants.keys()))
    bm25_class = bm25_variants[selected_variant_name]

    # --- Filter sumber berita ---
    if selected_source != "Semua":
        df = df[df["source"].str.contains(selected_source, case=False, na=False)]


    # --- Jalankan BM25 ---
    bm25 = bm25_class(df["Content"].tolist(), tokenizer=simple_tokenizer)

    if query:
        jumlah_dokumen = min(jumlah_dokumen, len(df))
        docs, scores = bm25.get_top_n(query, df["Content"].tolist(), n=jumlah_dokumen)
        result_indices = [df[df["Content"] == doc].index[0] for doc in docs]
        results = df.loc[result_indices].copy()
        results["bm25_score"] = scores

        for _, row in results.iterrows():
            tags = ", ".join([str(row[f"tag{i}"]) for i in range(1, 6) if pd.notna(row.get(f"tag{i}"))])
            st.markdown(f"""
                <div class='result-card'>
                    <div class='title'>{row['Judul']}</div>
                    <div class='info'>🗞️ <b>Sumber:</b> {row['source']} &nbsp;&nbsp; 🕒 <b>Waktu:</b> {row['Waktu']}</div>
                    <div class='info' style='margin-top:10px; line-height:1.6;'>{row["Content"][:500]}...</div>
                    {"<div class='info'>🏷️ <b>Tags:</b> " + tags + "</div>" if tags else ""}
                    <div class='info'>📈 <span class='score'>Skor {selected_variant_name}: {row['bm25_score']:.4f}</span></div>
                    <div class='info'>🔗 <a href='{row['Link']}' target='_blank'>Buka Artikel</a></div>
                </div>
            """, unsafe_allow_html=True)

# --- Run App ---
if __name__ == "__main__":
    main()