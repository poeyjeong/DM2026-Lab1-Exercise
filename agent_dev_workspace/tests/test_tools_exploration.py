import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity_tool():
    session = SessionState()
    
    # 1. Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM to populate session.feature_matrix and session.artifacts["count_vectorizer"]
    dtm_tools = {t.name: t for t in make_dtm_tools(session)}
    build_dtm = dtm_tools["build_dtm_tool"]
    build_dtm.invoke({})
    
    # 3. Instantiate cosine_similarity_tool
    exploration_tools = {t.name: t for t in make_exploration_tools(session)}
    cosine_tool = exploration_tools["cosine_similarity_tool"]
    
    # Test case 1: Doc 0 vs Doc 1 (both catA, near-identical text) -> ~0.9847
    res_0_1 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 1})
    assert pytest.approx(res_0_1["similarity"], abs=1e-3) == 0.9847
    
    # Test case 2: Doc 0 vs Doc 3 (catA vs catB, share only "always") -> ~0.0909
    res_0_3 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 3})
    assert pytest.approx(res_0_3["similarity"], abs=1e-3) == 0.0909
    
    # Test case 3: Doc 3 vs Doc 4 (both catB, near-identical text) -> ~0.9847
    res_3_4 = cosine_tool.invoke({"doc_idx_1": 3, "doc_idx_2": 4})
    assert pytest.approx(res_3_4["similarity"], abs=1e-3) == 0.9847
    
    # Test case 4: Doc 0 vs Doc 7 (catA vs empty-text row) -> 0.0
    res_0_7 = cosine_tool.invoke({"doc_idx_1": 0, "doc_idx_2": 7})
    assert pytest.approx(res_0_7["similarity"], abs=1e-3) == 0.0

def test_feature_correlation_matrix_tool():
    # Create a fresh session and load dataset using premade load_dataset_tool
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM first since feature_correlation_matrix_tool operates on the global DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Load exploration tools under test
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    # Invoke feature correlation matrix tool
    result = feat_corr_tool.invoke({})
    
    # Verify result shape and content against known answer #19
    assert "correlation_matrix" in result
    assert "feature_names" in result
    
    corr_matrix = np.array(result["correlation_matrix"])
    # sample_fixture has 5 terms, so matrix should be 5x5
    assert corr_matrix.shape == (5, 5)
    
    # Diagonal should be all 1.0
    np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-4)
    
    # Check specific known correlation pairs from TEST_FIXTURE.md #19
    feature_names = result["feature_names"]
    assert len(feature_names) == 5
    
    # Expected known answer matrix values from TEST_FIXTURE.md #19:
    # alpha, gamma, delta, beta, always
    expected_matrix = np.array([
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])
    
    np.testing.assert_allclose(corr_matrix, expected_matrix, atol=1e-3)
    
    # Verify session.pending_figure is set
    assert session.pending_figure is not None
