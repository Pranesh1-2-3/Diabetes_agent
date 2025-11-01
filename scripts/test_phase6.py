import json
import re
from pathlib import Path
from myservices.agent import analyze_case

# Example test case
TEST_CASES = {
    "case1": {
        "patient_data": {
            "age": 52,
            "gender": "male",
            "bmi": 29.4,
            "fasting_glucose": 132,
            "hba1c": 7.1,
        },
        "classifier_output": {
            "prediction": "likely_diabetes",
            "probabilities": {"diabetes": 0.88, "no_diabetes": 0.12},
        },
        "retrieved_chunks": [
            {
                "doc_id": "guideline1",
                "page_num": 12,
                "text": "Patients with fasting glucose >126 mg/dL or HbA1c >6.5% meet diagnostic criteria for diabetes.",
            }
        ],
        "memory": [
            {"date": "2023-09-01", "note": "Previous check-up showed borderline glucose levels."}
        ],
    }
}

def extract_json(text: str):
    """
    Extract the first valid JSON object from the model response.
    Removes ```json fences and ignores extra prose.
    """
    if not text:
        raise ValueError("Empty response from model")

    # Remove code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find the first {...} JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON: {e}\nRaw block:\n{match.group(0)}")

    raise ValueError(f"No JSON object found in response:\n{text[:500]}...")

def run_test_case(case_name: str, case_data: dict):
    """Run a single test case and return the agent response."""
    try:
        raw_response = analyze_case(
            patient_json=case_data["patient_data"],
            classifier_output=case_data["classifier_output"],
            retrieved_chunks=case_data["retrieved_chunks"],
            memory=case_data.get("memory"),
            model="llama-3.1-8b-instant"
        )

        # Ensure response is valid JSON
        response = extract_json(raw_response) if isinstance(raw_response, str) else raw_response

        print(f"\n=== {case_name.upper()} ===")
        print(json.dumps(response, indent=2))  # pretty JSON output

        return response

    except Exception as e:
        print(f"\n[ERROR] Test case {case_name} failed: {e}")
        return None

def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for case_name, case_data in TEST_CASES.items():
        response = run_test_case(case_name, case_data)

        if response:
            # Save structured JSON output
            output_file = results_dir / f"{case_name}_groq_output.json"
            with open(output_file, "w") as f:
                json.dump(response, f, indent=2)

            print(f"✅ Saved results for {case_name} to {output_file}")

if __name__ == "__main__":
    main()
