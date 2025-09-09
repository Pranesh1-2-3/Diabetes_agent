"""LLM-based medical analysis agent using Hugging Face Transformers."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from transformers import pipeline, set_seed

class Source(BaseModel):
    doc_id: str
    page: int
    quote: str
    relevance: str

class AnalysisResponse(BaseModel):
    conclusion: str
    reasoning: List[str]
    next_steps: List[str]
    sources: List[Source]

class DiabetesAgentHF:
    """Agent for analyzing diabetes cases using Hugging Face."""

    def __init__(self, model_name: str = "tiiuae/falcon-7b-instruct", max_input_length: int = 1024):
        self.model_name = model_name
        self.max_input_length = max_input_length

        # Hugging Face text-generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device=0,   # 0 for first GPU, -1 for CPU
            max_length=1024,
            do_sample=True,
            temperature=0.1
        )

        # Load prompts
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "diabetes_analysis.json", encoding="utf-8") as f:
            self.prompts = json.load(f)

    def _build_prompt(
        self,
        patient_data: Dict[str, Any],
        classifier_output: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        memory: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        chunks_text = ""
        for chunk in retrieved_chunks[:3]:  # limit number of chunks
            chunks_text += f"\nDocument: {chunk['doc_id']}, Page: {chunk['page_num']}\n"
            chunks_text += f"Content: {chunk['text'][:500]}...\n"

        memory_text = ""
        if memory:
            memory_text = "\n".join(f"Date: {entry['date']}\n{entry['note'][:200]}..." for entry in memory)

        prompt = self.prompts["user"]["template"].format(
            patient_json=json.dumps(patient_data, indent=2)[:500],
            classifier_output=json.dumps(classifier_output, indent=2),
            retrieved_chunks=chunks_text,
            memory=memory_text or "No previous history available."
        )

        # truncate prompt if too long
        if len(prompt) > self.max_input_length:
            prompt = prompt[:self.max_input_length] + "\n... [TRUNCATED]"

        return prompt

    def analyze(
        self,
        patient_data: Dict[str, Any],
        classifier_output: Dict[str, Any],
        retrieved_chunks: List[Dict[str, Any]],
        memory: Optional[List[Dict[str, Any]]] = None
    ) -> AnalysisResponse:
        prompt = self._build_prompt(patient_data, classifier_output, retrieved_chunks, memory)

        # Generate text from Hugging Face model
        output = self.generator(prompt, max_new_tokens=512)
        output_text = output[0]["generated_text"]

        # Parse JSON output
        try:
            result = json.loads(output_text)
            return AnalysisResponse(**result)
        except Exception as e:
            raise ValueError(f"Invalid HF response format: {e}")

    def evaluate_response(self, response: AnalysisResponse) -> Dict[str, Any]:
        return {
            "num_sources": len(response.sources),
            "has_quotes": all(len(s.quote) > 0 for s in response.sources),
            "reasoning_steps": len(response.reasoning),
            "has_next_steps": len(response.next_steps) > 0
        }
