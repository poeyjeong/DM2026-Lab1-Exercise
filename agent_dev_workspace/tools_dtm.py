from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session: Any) -> List[Any]:
    """
    Factory function returning tools for document-term matrix (DTM) processing.
    """

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        lowercase: bool = True
    ) -> Dict[str, Any]:
        """
        Builds a Document-Term Matrix (DTM) from text documents in session.dataframe.

        Converts raw text into a sparse feature matrix of token counts using CountVectorizer.
        Updates session.feature_matrix, session.feature_names, and stores the vectorizer 
        at session.artifacts["count_vectorizer"].

        Args:
            max_features (Optional[int]): Maximum number of terms to keep in vocabulary, ranked by frequency.
            ngram_range (List[int]): Lower and upper boundary of range of n-grams to extract, e.g. [1, 1].
            lowercase (bool): Whether to convert all characters to lowercase before tokenizing.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, document count, term count, 
                            matrix sparsity metrics, and configuration details.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        texts = session.dataframe["text"].fillna("").astype(str).tolist()

        ngram_tuple = tuple(ngram_range) if ngram_range and len(ngram_range) == 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
            lowercase=lowercase
        )

        dtm = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")

        args = {
            "max_features": max_features,
            "ngram_range": list(ngram_tuple),
            "lowercase": lowercase
        }

        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "max_features": max_features,
            "ngram_range": list(ngram_tuple),
            "lowercase": lowercase
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Aggregates term frequencies across all documents in the document-term matrix.

        Reads session.feature_matrix and session.feature_names, sums counts per term across
        all documents, and returns term frequency totals sorted descending.

        Args:
            top_n (Optional[int]): If specified, limits the output to top N most frequent terms.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, total_terms, total_frequency,
                            and term_frequencies map.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."}

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        terms = list(session.feature_names)

        df_tf = pd.DataFrame({"term": terms, "frequency": counts})
        df_tf["frequency"] = df_tf["frequency"].astype(int)
        df_tf = df_tf.sort_values(by="frequency", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("tf")

        if top_n is not None and top_n > 0:
            df_summary = df_tf.head(top_n)
        else:
            df_summary = df_tf

        term_freqs = dict(zip(df_summary["term"], df_summary["frequency"]))

        summary = {
            "result_id": result_id,
            "total_terms": len(terms),
            "returned_terms": len(term_freqs),
            "total_frequency": int(counts.sum()),
            "term_frequencies": term_freqs
        }

        args = {"top_n": top_n}

        session.store_result("term_frequency_tool", args, summary, full_report=df_tf)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """
        Generates a heatmap visualization for a positional slice of the Document-Term Matrix (DTM).

        Reads session.feature_matrix and session.feature_names, takes a positional slice 
        [0:n_documents, 0:n_terms], builds a Seaborn heatmap, and sets session.pending_figure.

        Args:
            n_terms (int): Number of terms (columns) to include from vocabulary order. Defaults to 20.
            n_documents (int): Number of documents (rows) to include from dataset row order. Defaults to 20.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, sliced shape, selected terms,
                            and the raw matrix slice values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "session.feature_matrix or session.feature_names is None. Run build_dtm_tool first."}

        total_docs, total_vocab = session.feature_matrix.shape
        actual_docs = min(n_documents, total_docs)
        actual_terms = min(n_terms, total_vocab)

        dtm_slice = session.feature_matrix[0:actual_docs, 0:actual_terms].toarray()
        selected_terms = list(session.feature_names[0:actual_terms])
        doc_indices = [f"Doc {i}" for i in range(actual_docs)]

        # Generate Seaborn heatmap plot
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.8), max(4, actual_docs * 0.5)))
        sns.heatmap(
            dtm_slice,
            xticklabels=selected_terms,
            yticklabels=doc_indices,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure for notebook/chat display
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")

        df_slice = pd.DataFrame(dtm_slice, index=doc_indices, columns=selected_terms)

        matrix_list = dtm_slice.tolist()

        summary = {
            "result_id": result_id,
            "sliced_shape": [actual_docs, actual_terms],
            "terms": selected_terms,
            "matrix_slice": matrix_list
        }

        args = {"n_terms": n_terms, "n_documents": n_documents}

        session.store_result("dtm_heatmap_tool", args, summary, full_report=df_slice)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
