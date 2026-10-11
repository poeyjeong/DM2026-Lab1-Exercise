from typing import List, Optional, Dict, Any
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from langchain_core.tools import tool

# Ensure punkt tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def make_tools(session: Any) -> List[Any]:
    """
    Factory function returning data-level tools, including tokenize_tool, list_files_tool, 
    inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, and describe_data_tool.
    """

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """
        Tokenizes each document's text in session.dataframe into unigrams.

        Uses nltk.word_tokenize on lowercased document text. Adds a 'unigrams' column
        to session.dataframe, stores the token lists in session.artifacts["tokens"],
        and returns a summary of the tokenization process.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, total_documents tokenized,
                            and sample tokens.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        # Tokenize each document: lowercase and apply word_tokenize
        token_lists = []
        for text in session.dataframe["text"].fillna("").astype(str):
            if not text.strip():
                token_lists.append([])
            else:
                token_lists.append(nltk.word_tokenize(text.lower()))

        session.dataframe["unigrams"] = token_lists
        session.artifacts["tokens"] = token_lists

        n_docs = len(token_lists)
        sample_tokens = token_lists[:3] if n_docs >= 3 else token_lists

        result_id = session.next_result_id("tokenize")
        args = {}

        summary = {
            "result_id": result_id,
            "n_documents_tokenized": n_docs,
            "sample_tokens": sample_tokens
        }

        session.store_result("tokenize_tool", args, summary, full_report=session.dataframe)
        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ""
    ) -> Dict[str, Any]:
        """
        Lists files and subdirectories in a specified workspace or repository subdirectory.

        Helps discover dataset paths and available files before loading a dataset.

        Args:
            subdirectory (str): Subdirectory path relative to the repo root to list. Defaults to root ("").

        Returns:
            Dict[str, Any]: A summary dict containing result_id, subdirectory, directories list, and files list.
        """
        target_dir = subdirectory.strip()
        if target_dir and not os.path.exists(target_dir):
            return {"error": f"Directory not found: {target_dir}"}

        path_to_list = target_dir if target_dir else "."
        try:
            entries = os.listdir(path_to_list)
        except Exception as e:
            return {"error": f"Failed to list directory {path_to_list}: {str(e)}"}

        directories = sorted([e for e in entries if os.path.isdir(os.path.join(path_to_list, e)) and not e.startswith('.')])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(path_to_list, e)) and not e.startswith('.')])

        result_id = session.next_result_id("list_files")
        args = {"subdirectory": subdirectory}

        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", args, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5
    ) -> Dict[str, Any]:
        """
        Peeks at the first few rows of session.dataframe to inspect dataset contents.

        Args:
            n_rows (int): Number of rows to preview. Defaults to 5.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, total_rows, columns, and preview records.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        total_rows = len(session.dataframe)
        columns = list(session.dataframe.columns)
        df_head = session.dataframe.head(n_rows)
        preview = df_head.to_dict(orient="records")

        result_id = session.next_result_id("inspect")
        args = {"n_rows": n_rows}

        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": columns,
            "preview": preview
        }

        session.store_result("inspect_data_tool", args, summary, full_report=df_head)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """
        Checks session.dataframe for missing or empty text values.

        Identifies rows where the text column is NaN, None, empty, or whitespace-only.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, total_rows, and n_missing count.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        df = session.dataframe
        # Check for missing (NaN/None) or empty/whitespace strings
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        n_missing = int(is_missing.sum())
        total_rows = len(df)

        result_id = session.next_result_id("check_missing")
        args = {}

        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "n_missing": n_missing
        }

        session.store_result("check_missing_tool", args, summary)
        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False
    ) -> Dict[str, Any]:
        """
        Checks for (and optionally drops) duplicate documents in session.dataframe.

        When dropping (drop=True), uses keep=False to remove all copies of duplicated rows,
        matching Master's X.drop_duplicates(keep=False, inplace=True) exactly.

        Args:
            drop (bool): Whether to drop duplicate rows in place. Defaults to False.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, n_duplicates count, dropped status, and remaining_rows.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        df = session.dataframe
        # Identify duplicates across text column
        dup_mask = df.duplicated(subset=["text"], keep=False)
        n_duplicates = int(dup_mask.sum())

        dropped = False
        if drop and n_duplicates > 0:
            df.drop_duplicates(subset=["text"], keep=False, inplace=True)
            dropped = True

        remaining_rows = len(df)

        result_id = session.next_result_id("check_duplicates")
        args = {"drop": drop}

        summary = {
            "result_id": result_id,
            "n_duplicates": n_duplicates,
            "dropped": dropped,
            "remaining_rows": remaining_rows
        }

        session.store_result("check_duplicates_tool", args, summary, full_report=df)
        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Subsamples the working DataFrame session.dataframe to a random subset of rows.

        Args:
            n (int): Number of rows to sample. Defaults to 5.
            random_state (Optional[int]): Random seed for reproducible sampling. Defaults to None.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, sampled_rows count, and sample indices.
        """
        if session.dataframe is None:
            return {"error": "session.dataframe is None. Load a dataset first."}

        df = session.dataframe
        actual_n = min(n, len(df))
        sampled_df = df.sample(n=actual_n, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]

        session.dataframe = sampled_df

        result_id = session.next_result_id("sample")
        args = {"n": n, "random_state": random_state}

        summary = {
            "result_id": result_id,
            "sampled_rows": len(sampled_df),
            "sampled_indices": sampled_indices
        }

        session.store_result("sample_data_tool", args, summary, full_report=sampled_df)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes descriptive statistics on text length (character count) overall and per category,
        and generates a box plot set on session.pending_figure.

        Returns:
            Dict[str, Any]: A summary dict containing result_id, overall statistics, and per-category statistics.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "session.dataframe is missing or does not contain a 'text' column."}

        df = session.dataframe.copy()
        df["text_length"] = df["text"].fillna("").astype(str).apply(len)

        # Overall describe stats
        overall_desc = df["text_length"].describe().to_dict()

        # Per category describe stats
        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        per_category_desc = {}
        if category_col:
            for cat, group in df.groupby(category_col):
                per_category_desc[str(cat)] = group["text_length"].describe().to_dict()

        # Generate box plot by category
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_col:
            sns.boxplot(data=df, x=category_col, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=df["text_length"], ax=ax)
            ax.set_title("Text Length Distribution")
            ax.set_xlabel("Dataset")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        args = {}

        summary = {
            "result_id": result_id,
            "overall_stats": {k: float(v) for k, v in overall_desc.items()},
            "per_category_stats": {
                cat: {k: float(v) for k, v in stats.items()} 
                for cat, stats in per_category_desc.items()
            }
        }

        session.store_result("describe_data_tool", args, summary, full_report=df)
        return summary

    return [
        tokenize_tool, 
        list_files_tool, 
        inspect_data_tool, 
        check_missing_tool, 
        check_duplicates_tool, 
        sample_data_tool, 
        describe_data_tool
    ]
