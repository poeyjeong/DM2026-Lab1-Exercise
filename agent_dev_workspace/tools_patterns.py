import os
import tempfile
import pandas as pd
import numpy as np
from typing import Literal, Optional, List, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PAMI.extras.convert.DF2DB import DF2DB
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent patterns for a specific category using PAMI after filtering vocabulary.
        
        Args:
            category_name: Which category to mine.
            filtering_method: Vocabulary filter ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for fpgrowth and maxfpgrowth).
            k: Number of top-k patterns to keep (required for topk).
        """
        if session.dataframe is None or session.dataframe.empty:
            result_id = session.next_result_id("mine_patterns")
            summary = {"error": "No dataset loaded in session.", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary
        
        df = session.dataframe
        if "category_name" not in df.columns:
            result_id = session.next_result_id("mine_patterns")
            summary = {"error": "Dataset missing category_name column.", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary
        
        # 1. Filter dataset to category_name and build fresh category-local DTM
        cat_df = df[df["category_name"] == category_name]
        texts = cat_df["text"].fillna("").tolist()
        
        if not texts or all(t.strip() == "" for t in texts):
            result_id = session.next_result_id("mine_patterns")
            summary = {"patterns": [], "note": "no documents or empty text for category", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary
            
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_counts = vectorizer.fit_transform(texts)
        except ValueError:
            result_id = session.next_result_id("mine_patterns")
            summary = {"patterns": [], "note": "no terms survived stopword removal", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary
            
        feature_names = np.array(vectorizer.get_feature_names_out())
        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {"patterns": [], "note": "no terms survived stopword removal", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # 2. Apply filtering method
        dense_counts = X_counts.toarray()
        kept_indices = []
        
        if filtering_method == "variance":
            variances = np.var(dense_counts, axis=0)
            if len(variances) > 0:
                p5, p95 = np.percentile(variances, [5, 95])
                kept_indices = [i for i, v in enumerate(variances) if p5 <= v <= p95]
        elif filtering_method == "term_frequency":
            col_sums = np.sum(dense_counts, axis=0)
            if len(col_sums) > 0:
                p5, p95 = np.percentile(col_sums, [5, 95])
                kept_indices = [i for i, s in enumerate(col_sums) if p5 <= s <= p95]
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(X_counts).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            if len(mean_tfidf) > 0:
                p20 = np.percentile(mean_tfidf, 20)
                kept_indices = [i for i, val in enumerate(mean_tfidf) if val >= p20]
                
        if not kept_indices:
            result_id = session.next_result_id("mine_patterns")
            summary = {"patterns": [], "note": "no terms survived filtering", "result_id": result_id}
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary
            
        filtered_counts = dense_counts[:, kept_indices]
        filtered_features = feature_names[kept_indices]
        
        # Build DataFrame for DF2DB
        filtered_df = pd.DataFrame(filtered_counts, columns=filtered_features)
        
        # 3. Convert to transactional database via PAMI DF2DB passing DataFrame directly with condition and threshold
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_db:
            tmp_db_path = tmp_db.name
            
        try:
            obj = DF2DB(filtered_df)
            obj.convert2TransactionalDatabase(tmp_db_path, ">", 0)
            
            # 4. Mine patterns using requested algorithm
            patterns_list = []
            if algorithm == "fpgrowth":
                if min_sup is None:
                    min_sup = 1
                miner = FPGrowth(tmp_db_path, minSup=min_sup, sep="\t")
                miner.mine()
                for pat, sup in miner.getPatterns().items():
                    patterns_list.append({"pattern": list(pat) if isinstance(pat, tuple) else [pat], "support": sup})
            elif algorithm == "topk":
                if k is None:
                    k = 1
                miner = FAE(tmp_db_path, k=k, sep="\t")
                miner.mine()
                for pat, sup in miner.getPatterns().items():
                    patterns_list.append({"pattern": list(pat) if isinstance(pat, tuple) else [pat], "support": sup})
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    min_sup = 1
                miner = MaxFPGrowth(tmp_db_path, minSup=min_sup, sep="\t")
                miner.mine()
                for pat, sup in miner.getPatterns().items():
                    patterns_list.append({"pattern": list(pat) if isinstance(pat, tuple) else [pat], "support": sup})
        finally:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)
                
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list
        
        result_id = session.next_result_id("mine_patterns")
        summary = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": patterns_list,
            "n_patterns": len(patterns_list),
            "result_id": result_id
        }
        
        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)
        
        return summary

    return [mine_patterns_tool]
