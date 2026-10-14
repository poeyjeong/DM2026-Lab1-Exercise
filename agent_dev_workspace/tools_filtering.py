import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function returning filtering tools for the agent pipeline.
    """
    
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> dict:
        """
        Filters terms from the global document-term matrix by variance,
        identifying near-constant terms for removal.
        
        Args:
            threshold (float): Minimum variance required for a term to be kept.
            
        Returns:
            dict: Summary of filtering results including kept/removed counts and result ID.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please build DTM first."}
        
        matrix_dense = session.feature_matrix.toarray()
        variances = np.var(matrix_dense, axis=0)
        
        report_data = []
        for term, var in zip(session.feature_names, variances):
            report_data.append({"term": term, "variance": float(var)})
            
        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values(by="variance", ascending=False).reset_index(drop=True)
        
        kept_mask = df_report["variance"] >= threshold
        kept_terms = df_report[kept_mask]["term"].tolist()
        removed_terms = df_report[~kept_mask]["term"].tolist()
        
        result_id = session.next_result_id("variance_filter")
        
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "kept_count": len(kept_terms),
            "removed_count": len(removed_terms),
            "kept_terms_preview": kept_terms[:5]
        }
        
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )
        
        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> dict:
        """
        Computes Pearson correlation between each term's count vector in the global
        document-term matrix and a target class binary indicator (one-vs-rest).
        
        Args:
            target_class (str): The specific category name to compute correlation against.
            
        Returns:
            dict: Summary of Pearson correlation results including result ID and preview of top correlations.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {"error": "Document-term matrix or dataframe not found. Please load dataset and build DTM first."}
        
        if "category_name" not in session.dataframe.columns:
            return {"error": "Category names not found in session dataframe."}
        
        labels = session.dataframe["category_name"].values
        target_vector = np.array([1 if label == target_class else 0 for label in labels])
        
        matrix_dense = session.feature_matrix.toarray()
        
        report_data = []
        correlations = {}
        
        for i, term in enumerate(session.feature_names):
            term_vector = matrix_dense[:, i]
            if np.std(term_vector) == 0 or np.std(target_vector) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(term_vector, target_vector)
                if np.isnan(r):
                    r = 0.0
            
            report_data.append({"term": term, "pearson_r": float(r)})
            correlations[term] = float(r)
            
        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values(by="pearson_r", ascending=False).reset_index(drop=True)
        
        result_id = session.next_result_id("pearson_filter")
        
        top_preview = df_report.head(5).to_dict(orient="records")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms_analyzed": len(session.feature_names),
            "top_correlations_preview": top_preview
        }
        
        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class},
            summary=summary,
            full_report=df_report
        )
        
        return summary

    @tool
    def spearman_filter_tool(target_class: str) -> dict:
        """
        Computes Spearman rank correlation between each term's count vector in the global
        document-term matrix and a target class binary indicator (one-vs-rest).
        
        Args:
            target_class (str): The specific category name to compute rank correlation against.
            
        Returns:
            dict: Summary of Spearman correlation results including result ID and preview of top correlations.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            return {"error": "Document-term matrix or dataframe not found. Please load dataset and build DTM first."}
        
        if "category_name" not in session.dataframe.columns:
            return {"error": "Category names not found in session dataframe."}
        
        labels = session.dataframe["category_name"].values
        target_vector = np.array([1 if label == target_class else 0 for label in labels])
        
        matrix_dense = session.feature_matrix.toarray()
        
        report_data = []
        
        for i, term in enumerate(session.feature_names):
            term_vector = matrix_dense[:, i]
            if np.std(term_vector) == 0 or np.std(target_vector) == 0:
                r = 0.0
            else:
                r, _ = spearmanr(term_vector, target_vector)
                if np.isnan(r):
                    r = 0.0
            
            report_data.append({"term": term, "spearman_r": float(r)})
            
        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values(by="spearman_r", ascending=False).reset_index(drop=True)
        
        result_id = session.next_result_id("spearman_filter")
        
        top_preview = df_report.head(5).to_dict(orient="records")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms_analyzed": len(session.feature_names),
            "top_correlations_preview": top_preview
        }
        
        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class},
            summary=summary,
            full_report=df_report
        )
        
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
