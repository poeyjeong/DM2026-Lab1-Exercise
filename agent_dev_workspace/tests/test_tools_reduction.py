import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def setup_session_and_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build feature matrix for reduction tests
    vectorizer = CountVectorizer()
    texts = session.dataframe["text"].fillna("").tolist()
    X = vectorizer.fit_transform(texts)
    session.feature_matrix = X
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    
    tools = {t.name: t for t in make_tools(session)}
    return session, tools

def test_reduce_dimensions_pca():
    session, tools = setup_session_and_dtm()
    tool = tools["reduce_dimensions_tool"]
    
    res = tool.invoke({"method": "pca"})
    
    assert session.pending_figure is not None
    assert "explained_variance_ratio" in res
    
    expected_evr = [0.7532, 0.1965]
    assert pytest.approx(res["explained_variance_ratio"], abs=1e-3) == expected_evr
    
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ]
    assert np.allclose(res["coordinates_preview"], expected_coords, atol=1e-3)

def test_reduce_dimensions_tsne():
    session, tools = setup_session_and_dtm()
    tool = tools["reduce_dimensions_tool"]
    
    res = tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    
    assert session.pending_figure is not None
    
    expected_coords = res["coordinates_preview"]
    assert np.allclose(res["coordinates_preview"], expected_coords, atol=1e-3)

def test_reduce_dimensions_umap():
    session, tools = setup_session_and_dtm()
    tool = tools["reduce_dimensions_tool"]
    
    res = tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    
    assert session.pending_figure is not None
    
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ]
    assert np.allclose(res["coordinates_preview"], expected_coords, atol=1e-2)

def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = {t.name: t for t in make_tools(session)}
    tool = tools["binarize_labels_tool"]
    
    res = tool.invoke({"positive_class": "catA"})
    
    assert "result_id" in res
    assert res["positive_class"] == "catA"
    assert res["positive_count"] == 4
    assert res["negative_count"] == 4
    assert "label_bin" in session.dataframe.columns
    assert int(session.dataframe["label_bin"].sum()) == 4
