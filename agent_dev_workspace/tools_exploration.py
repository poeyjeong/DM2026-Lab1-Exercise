from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function returning exploration tools such as cosine_similarity_tool and feature_correlation_matrix_tool.
    """
    
    def cosine_similarity_tool(
        doc_idx_1: Optional[int] = None,
        doc_idx_2: Optional[int] = None,
        text_1: Optional[str] = None,
        text_2: Optional[str] = None
    ) -> dict:
        """
        Computes the cosine similarity between two documents using the existing fitted CountVectorizer
        from session.artifacts["count_vectorizer"] and/or session.feature_matrix.

        Args:
            doc_idx_1: Index of the first document in session.feature_matrix (optional).
            doc_idx_2: Index of the second document in session.feature_matrix (optional).
            text_1: Raw text of the first document (optional, overrides doc_idx_1 if provided).
            text_2: Raw text of the second document (optional, overrides doc_idx_2 if provided).
        """
        vectorizer = session.artifacts.get("count_vectorizer")
        if vectorizer is None:
            return {"error": "Count vectorizer not found in session.artifacts. Build DTM first."}

        # Helper to get vector representation for a source (index or text)
        def get_vector(idx, text):
            if text is not None:
                return vectorizer.transform([text])
            elif idx is not None:
                if session.feature_matrix is None:
                    return {"error": "feature_matrix is None."}
                if idx < 0 or idx >= session.feature_matrix.shape[0]:
                    return {"error": f"Document index {idx} out of bounds."}
                return session.feature_matrix[idx:idx+1]
            else:
                return None

        vec1 = get_vector(doc_idx_1, text_1)
        vec2 = get_vector(doc_idx_2, text_2)

        if isinstance(vec1, dict):
            return vec1
        if isinstance(vec2, dict):
            return vec2
        if vec1 is None or vec2 is None:
            return {"error": "Must specify either document indices or text strings for both inputs."}

        sim_matrix = cosine_similarity(vec1, vec2)
        similarity = float(sim_matrix[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "similarity": round(similarity, 4),
            "doc_idx_1": doc_idx_1,
            "doc_idx_2": doc_idx_2,
            "text_1": text_1,
            "text_2": text_2
        }

        args = {
            "doc_idx_1": doc_idx_1,
            "doc_idx_2": doc_idx_2,
            "text_1": text_1,
            "text_2": text_2
        }
        session.store_result("cosine_similarity_tool", args, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Computes the Pearson correlation matrix over the top 20 terms by variance from the global DTM, generates an annotated heatmap, and stores the result.

        Args:
            None
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Feature matrix and feature names must be built first using build_dtm_tool."}

        # Convert sparse feature matrix to dense array
        X = session.feature_matrix.toarray()
        feature_names = session.feature_names

        # Compute variance for each term (columns)
        variances = np.var(X, axis=0)

        # Get top 20 terms by variance (or all if fewer than 20)
        n_terms = min(20, X.shape[1])
        top_indices = np.argsort(variances)[::-1][:n_terms]

        selected_names = [feature_names[i] for i in top_indices]
        X_top = X[:, top_indices]

        # Compute Pearson correlation matrix across terms (rowvar=False since terms are columns)
        corr_matrix = np.corrcoef(X_top, rowvar=False)

        # Build heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=selected_names,
            yticklabels=selected_names,
            cmap="coolwarm",
            ax=ax
        )
        plt.tight_layout()
        session.pending_figure = fig

        # Prepare summary (converting numpy array to nested list for JSON serializability)
        result_id = session.next_result_id("feature_correlation_matrix")
        summary = {
            "result_id": result_id,
            "feature_names": selected_names,
            "correlation_matrix": corr_matrix.tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [
        tool(cosine_similarity_tool),
        feature_correlation_matrix_tool
    ]
