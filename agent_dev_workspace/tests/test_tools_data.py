import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Get and invoke tokenize_tool
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    result = tokenize_tool.invoke({})
    
    assert "result_id" in result
    assert result["n_documents_tokenized"] == 8
    
    # Check tokens against TEST_FIXTURE.md known answer #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens

def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")
    
    # Test listing "agent_dev" subdirectory per request and known answer #11 style
    result = list_files_tool.invoke({"subdirectory": "agent_dev"})
    
    assert "result_id" in result
    assert "files" in result
    assert "sample_fixture.csv" in result["files"]

def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")
    
    result = inspect_data_tool.invoke({"n_rows": 3})
    
    assert "result_id" in result
    assert result["total_rows"] == 8
    assert "text" in result["columns"]
    assert "category_name" in result["columns"]
    assert len(result["preview"]) == 3

def test_check_missing_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")
    
    result = check_missing_tool.invoke({})
    
    assert "result_id" in result
    # TEST_FIXTURE.md Known Answer #2: Exactly 1 row has missing/empty text
    assert result["n_missing"] == 1

def test_check_duplicates_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")
    
    # TEST_FIXTURE.md Known Answer #3: Exactly 1 duplicate pair, dropping with keep=False leaves 6 rows
    result = check_duplicates_tool.invoke({"drop": True})
    
    assert "result_id" in result
    assert result["n_duplicates"] == 2  # Both copies of the duplicate pair are flagged by keep=False
    assert result["dropped"] is True
    assert result["remaining_rows"] == 6

def test_sample_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")
    
    # TEST_FIXTURE.md Known Answer #13: n=4, random_state=42 returns rows [1, 5, 0, 7]
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})
    
    assert "result_id" in result
    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]

def test_describe_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset fixture
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")
    
    result = describe_data_tool.invoke({})
    
    assert "result_id" in result
    # TEST_FIXTURE.md Known Answer #18: Overall mean=19.875, catA mean=17.25, catB mean=22.5
    assert result["overall_stats"]["count"] == 8
    assert result["overall_stats"]["mean"] == pytest.approx(19.875, abs=1e-3)
    assert result["overall_stats"]["min"] == 0
    assert result["overall_stats"]["max"] == 30
    
    assert "catA" in result["per_category_stats"]
    assert result["per_category_stats"]["catA"]["mean"] == pytest.approx(17.25, abs=1e-3)
    
    assert "catB" in result["per_category_stats"]
    assert result["per_category_stats"]["catB"]["mean"] == pytest.approx(22.5, abs=1e-3)
    
    assert session.pending_figure is not None
