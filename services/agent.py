# services/agent.py
import json
import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # load variables from .env
api_key = os.getenv("GROQ_API_KEY")

# Path to your prompt template
PROMPTS_PATH = Path(__file__).parent.parent / "prompts" / "diabetes_analysis.json"

class DiabetesAgent:
    """Agent class for diabetes analysis using Groq LLM."""
    
    def __init__(self, model="llama-3.1-8b-instant"):
        """Initialize the agent with Groq model."""
        self.model = model
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def build_prompt(self, patient_json, classifier_output, retrieved_chunks, memory):
        """Build the final prompt from template."""
        with open(PROMPTS_PATH) as f:
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
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Groq output is not valid JSON: {e}\nRaw output:\n{content}")

    def analyze_case(self, patient_json, classifier_output, retrieved_chunks, memory):
        """Main function to run the full agent pipeline with Groq."""
        system_msg, user_msg = self.build_prompt(patient_json, classifier_output, retrieved_chunks, memory)
        return self.run_groq(system_msg, user_msg)
