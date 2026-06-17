import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.multidomain_schema.v2.generate_fpg_dataset import (
    generate_comparative_order_sample,
    generate_logic_rules_sample
)

def test_generate_comparative_order_sample():
    """Test the procedural generator for comparative_order."""
    sample = generate_comparative_order_sample(num_objects=3)
    
    assert sample["domain"] == "comparative_order"
    assert "schema" in sample
    assert "solution" in sample
    assert "input_text" in sample
    
    # Check schema
    schema = sample["schema"]
    assert len(schema["objects"]) == 3
    assert len(schema["relations"]) > 0
    
    # Check solution
    assert len(sample["solution"]["order"]) == 3
    
    # Check text
    assert len(sample["input_text"]) > 10

def test_generate_logic_rules_sample():
    """Test the procedural generator for logic_rules."""
    sample = generate_logic_rules_sample(num_rules=4)
    
    assert sample["domain"] == "logic_rules"
    assert "schema" in sample
    assert "solution" in sample
    assert "input_text" in sample
    
    # Check schema
    schema = sample["schema"]
    assert len(schema["rules"]) == 4
    assert len(schema["facts"]) > 0
    assert "query" in schema
    
    # Check solution
    assert "answer" in sample["solution"]
    assert isinstance(sample["solution"]["answer"], bool)
    
    # Check text
    assert len(sample["input_text"]) > 10
