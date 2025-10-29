# services/agent.py
import json
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import os
from fastapi import HTTPException

load_dotenv()  # load variables from .env
api_key = os.getenv("GROQ_API_KEY")

# Path to your prompt template
PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "diabetes_analysis.json"

class DiabetesAgent:
    """Agent class for diabetes analysis using Groq LLM."""
    
    def __init__(self, model="llama-3.1-8b-instant", patient_centered=True):
        """Initialize the agent with Groq model."""
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.patient_centered = patient_centered
        self.prompt_file = "patient_centered_analysis.json" if patient_centered else "diabetes_analysis.json"

    def build_prompt(self, patient_json, classifier_output, retrieved_chunks, memory):
        """Build the final prompt from template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / self.prompt_file
        with open(prompt_path) as f:
            prompt_template = json.load(f)

        system_msg = prompt_template["system"]
        user_template = prompt_template["user"]["template"]

        user_msg = user_template.format(
            patient_json=json.dumps(patient_json, indent=2),
            classifier_output=json.dumps(classifier_output, indent=2),
            retrieved_chunks=json.dumps(retrieved_chunks, indent=2),
            memory=json.dumps(memory, indent=2),
        )

        return system_msg, user_msg


    def run_groq(self, system_msg, user_msg):
        """Run Groq with system and user messages."""
        messages = [
            {"role": "system", "content": system_msg["content"]},
            {"role": "user", "content": user_msg},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # The model is expected to return JSON (per your schema)
        content = response.choices[0].message.content.strip()
        if not content:
            raise HTTPException(status_code=500, detail="Groq returned empty response")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
                raise HTTPException(
        status_code=500,
        detail=f"Groq returned invalid JSON: {content}\n\nError: {str(e)}"
    )

    def analyze_case(self, patient_json, classifier_output, retrieved_chunks, memory):
        """Main function to run the full agent pipeline with Groq."""
        system_msg, user_msg = self.build_prompt(patient_json, classifier_output, retrieved_chunks, memory)
        return self.run_groq(system_msg, user_msg)
