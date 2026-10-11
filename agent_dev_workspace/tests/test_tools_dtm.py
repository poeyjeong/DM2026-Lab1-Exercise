import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    assert res["n_terms"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    assert res["n_documents"] == 8
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert pytest.approx(res["sparsity_pct"], abs=1e-3) == 47.5

    assert "count_vectorizer" in session.artifacts

def test_term_frequency_tool_known_answer_14():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({})

    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }

    assert res["total_terms"] == 5
    assert res["total_frequency"] == 27
    assert res["term_frequencies"] == expected_freqs

def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # 1. Test full matrix slice (n_terms=20, n_documents=20)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res_full["sliced_shape"] == [8, 5]
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert res_full["matrix_slice"] == expected_full_matrix
    assert session.pending_figure is not None

    # Reset pending figure for sliced test
    session.pending_figure = None

    # 2. Test top-left corner slice (n_terms=2, n_documents=3)
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_corner["sliced_shape"] == [3, 2]
    assert res_corner["terms"] == ["alpha", "always"]
    expected_corner_matrix = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert res_corner["matrix_slice"] == expected_corner_matrix
    assert session.pending_figure is not None
