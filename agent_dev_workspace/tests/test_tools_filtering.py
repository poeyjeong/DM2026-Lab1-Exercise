import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools
from tools_dtm import make_tools as make_dtm_tools

def test_variance_filter_tool_known_answer_5():
    """
    Tests variance_filter_tool against known answer #5 from TEST_FIXTURE.md
    using sample_fixture.csv.
    """
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    res = variance_filter_tool.invoke({"threshold": 0.15})
    
    full_report = session.results_store[res["result_id"]]["full_report"]
    variances_dict = dict(zip(full_report["term"], full_report["variance"]))
    assert variances_dict["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances_dict["gamma"] == pytest.approx(1.1094, abs=1e-3)
    assert variances_dict["delta"] == pytest.approx(0.2500, abs=1e-3)
    assert variances_dict["beta"] == pytest.approx(0.2344, abs=1e-3)
    assert variances_dict["always"] == pytest.approx(0.1094, abs=1e-3)
    
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]

def test_pearson_filter_tool_known_answer_6():
    """
    Tests pearson_filter_tool against known answer #6 from TEST_FIXTURE.md
    using sample_fixture.csv with target_class="catB".
    """
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    filtering_tools = make_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    res = pearson_filter_tool.invoke({"target_class": "catB"})
    
    corrs = res["correlations"]
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)

def test_spearman_filter_tool_known_answer_6():
    """
    Tests spearman_filter_tool against known answer #6 from TEST_FIXTURE.md
    using sample_fixture.csv with target_class="catB".
    """
    session = SessionState()
    
    # 1. Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # 2. Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # 3. Run spearman_filter_tool with target_class="catB"
    filtering_tools = make_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    res = spearman_filter_tool.invoke({"target_class": "catB"})
    
    # Check Spearman rank correlation against known answer #6:
    # alpha: -0.7500
    # always: 0.3780
    # beta: -0.7746
    # delta: 1.0000
    # gamma: 0.9363
    corrs = res["correlations"]
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)
