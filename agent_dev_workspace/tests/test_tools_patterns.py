import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def test_mine_patterns_fpgrowth_and_topk():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    
    # Known answer #17: fpgrowth with tfidf filtering
    res_catA_tfidf = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    
    patterns_a = res_catA_tfidf["patterns"]
    assert len(patterns_a) > 0
    # Pattern {{'alpha'}} with support 3
    alpha_pat = next((p for p in patterns_a if set(p["pattern"]) == {"alpha"}), None)
    assert alpha_pat is not None
    assert alpha_pat["support"] == 3
    
    res_catB_tfidf = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    
    patterns_b = res_catB_tfidf["patterns"]
    assert len(patterns_b) > 0
    gamma_pat = next((p for p in patterns_b if set(p["pattern"]) == {"gamma"}), None)
    assert gamma_pat is not None
    assert gamma_pat["support"] == 4
    
    # Known answer #17: variance and term_frequency filtering keep zero terms
    res_catA_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_catA_var["patterns"] == []
    assert res_catA_var.get("note") == "no terms survived filtering"
    
    # Known answer #20: topk algorithm
    res_topk = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert len(res_topk["patterns"]) == 1
    assert set(res_topk["patterns"][0]["pattern"]) == {"alpha"}
    assert res_topk["patterns"][0]["support"] == 3
    
    # Known answer #20: maxfpgrowth returns zero patterns
    res_max = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["patterns"] == []
