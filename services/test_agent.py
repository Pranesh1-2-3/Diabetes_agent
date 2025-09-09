"""Tests for the Hugging Face-based LLM analysis agent."""

import json
from pathlib import Path
import pytest
from .agent import DiabetesAgentHF, AnalysisResponse, Source

# Sample test data
PATIENT_DATA = {
    "age": 45,
    "gender": "F",
    "bmi": 32.4,
    "blood_pressure": "140/90",
    "glucose": 180
}

CLASSIFIER_OUTPUT = {
    "prediction": "high_risk",
    "probability": 0.85,
    "features": ["bmi", "glucose", "blood_pressure"]
}

RETRIEVED_CHUNKS = [
    {
        "doc_id": "WHO_Fact_Sheet",
        "page_num": 2,
        "text": "BMI >= 30 indicates obesity and increases diabetes risk."
    },
    {
        "doc_id": "IDF_Atlas",
        "page_num": 15,
        "text": "Fasting glucose >= 126 mg/dL indicates diabetes."
    }
]

@pytest.fixture
def agent():
    """Create Hugging Face agent instance for testing."""
    return DiabetesAgentHF(model_name="dummy-model")

def test_prompt_building(agent):
    """Test prompt template filling."""
    prompt = agent._build_prompt(
        PATIENT_DATA,
        CLASSIFIER_OUTPUT,
        RETRIEVED_CHUNKS
    )
    
    # Verify all components are included
    assert json.dumps(PATIENT_DATA)[:50] in prompt
    assert json.dumps(CLASSIFIER_OUTPUT)[:50] in prompt
    assert "WHO_Fact_Sheet" in prompt
    assert "IDF_Atlas" in prompt

def test_response_validation():
    """Test response model validation."""
    # Valid response
    response = AnalysisResponse(
        conclusion="Patient shows high risk factors",
        reasoning=["BMI indicates obesity", "Elevated glucose levels"],
        next_steps=["Schedule follow-up", "Dietary consultation"],
        sources=[
            Source(
                doc_id="WHO_Fact_Sheet",
                page=2,
                quote="BMI >= 30 indicates obesity",
                relevance="Confirms obesity diagnosis"
            )
        ]
    )
    assert isinstance(response, AnalysisResponse)
    
    # Invalid response should raise ValueError
    with pytest.raises(ValueError):
        AnalysisResponse(
            conclusion="test",
            reasoning=["test"],
            next_steps=[],  # Empty next steps
            sources=[]  # Empty sources
        )

def test_response_evaluation(agent):
    """Test quality metrics calculation."""
    response = AnalysisResponse(
        conclusion="Test conclusion",
        reasoning=["Step 1", "Step 2"],
        next_steps=["Next step 1"],
        sources=[
            Source(
                doc_id="test",
                page=1,
                quote="test quote",
                relevance="test"
            )
        ]
    )
    
    metrics = agent.evaluate_response(response)
    assert metrics["num_sources"] == 1
    assert metrics["has_quotes"] is True
    assert metrics["reasoning_steps"] == 2
    assert metrics["has_next_steps"] is True

# Integration test - safe, uses mocked HF pipeline
@pytest.mark.integration
def test_analyze_integration(monkeypatch):
    """Test full analysis flow with mocked Hugging Face pipeline."""
    # Mock HF generator output
    class MockPipeline:
        def __call__(self, prompt, max_new_tokens=512):
            return [{"generated_text": json.dumps({
                "conclusion": "test",
                "reasoning": ["test reasoning"],
                "next_steps": ["test step"],
                "sources": [{
                    "doc_id": "test_doc",
                    "page": 1,
                    "quote": "test quote",
                    "relevance": "test relevance"
                }]
            })}]

    # Patch the generator
    agent = DiabetesAgentHF(model_name="dummy-model")
    monkeypatch.setattr(agent, "generator", MockPipeline())

    response = agent.analyze(
        PATIENT_DATA,
        CLASSIFIER_OUTPUT,
        RETRIEVED_CHUNKS
    )

    assert isinstance(response, AnalysisResponse)
    assert len(response.sources) > 0
