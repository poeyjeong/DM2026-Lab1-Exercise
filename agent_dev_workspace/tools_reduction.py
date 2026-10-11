from typing import Optional, Literal, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def make_tools(session):
    """
    Factory function returning the dimensionality reduction and label binarization tools.
    """
    
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None
    ) -> dict:
        """
        Reduces the dimensionality of session.feature_matrix to 2D using PCA, t-SNE, or UMAP,
        generates a scatter plot colored by category, sets session.pending_figure, and stores the result.

        Args:
            method: The reduction algorithm to use ("pca", "tsne", or "umap").
            perplexity: Perplexity parameter for t-SNE (useful to lower for small test fixtures).
            n_neighbors: Number of neighbors parameter for UMAP.
            random_state: Seed for reproducibility.
        """
        if session.feature_matrix is None:
            return {"error": "feature_matrix is None. Build DTM first."}
        
        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = np.array(X)

        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=2, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = model.explained_variance_ratio_.tolist()
        elif method == "tsne":
            kwargs = {"n_components": 2, "perplexity": perplexity}
            if random_state is not None:
                kwargs["random_state"] = random_state
            model = TSNE(**kwargs)
            coords = model.fit_transform(X_dense)
        elif method == "umap":
            kwargs = {"n_components": 2, "n_neighbors": n_neighbors}
            if random_state is not None:
                kwargs["random_state"] = random_state
            model = umap.UMAP(**kwargs)
            coords = model.fit_transform(X_dense)
        else:
            return {"error": f"Unknown reduction method: {method}"}

        # Store coordinates in artifacts
        artifact_key = f"reduced_coords_{method}"
        session.artifacts[artifact_key] = coords

        # Build scatter plot and set session.pending_figure without closing the figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            hue = session.dataframe["category_name"]
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            hue = session.dataframe["category"]
        else:
            hue = np.zeros(coords.shape[0])

        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=hue,
            palette="Set1",
            ax=ax,
            s=100,
            alpha=0.8
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.legend(title="Category")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_samples": int(coords.shape[0]),
            "coordinates_preview": coords.tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args = {
            "method": method,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state
        }
        full_report = pd.DataFrame(coords, columns=["dim1", "dim2"])
        session.store_result("reduce_dimensions_tool", args, summary, full_report=full_report)

        return summary

    def binarize_labels_tool(positive_class: str) -> dict:
        """
        Creates a 'label_bin' column in session.dataframe (1 if category_name matches positive_class, else 0)
        and returns the positive and negative counts.

        Args:
            positive_class: The category string to treat as the positive class (1).
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe or category_name column missing."}

        # Create label_bin column
        session.dataframe["label_bin"] = (session.dataframe["category_name"] == positive_class).astype(int)
        
        positive_count = int((session.dataframe["label_bin"] == 1).sum())
        negative_count = int((session.dataframe["label_bin"] == 0).sum())

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "positive_class": positive_class,
            "positive_count": positive_count,
            "negative_count": negative_count
        }

        args = {"positive_class": positive_class}
        full_report = session.dataframe[["category_name", "label_bin"]].copy()
        session.store_result("binarize_labels_tool", args, summary, full_report=full_report)

        return summary

    from langchain_core.tools import tool
    return [
        tool(reduce_dimensions_tool),
        tool(binarize_labels_tool)
    ]
