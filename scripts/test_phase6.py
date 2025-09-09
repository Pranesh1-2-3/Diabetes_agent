import json
from pathlib import Path

from services.agent import DiabetesAgentHF

# Example test cases (expand as needed)
TEST_CASES = {
    "case1": {
        "patient_data": {
            "age": 52,
            "gender": "male",
            "bmi": 29.4,
            "fasting_glucose": 132,
            "hba1c": 7.1
        },
        "classifier_output": {
            "prediction": "likely_diabetes",
            "probabilities": {"diabetes": 0.88, "no_diabetes": 0.12}
        },
        "retrieved_chunks": [
            {
                "doc_id": "guideline1",
                "page_num": 12,
                "text": "Patients with fasting glucose >126 mg/dL or HbA1c >6.5% meet diagnostic criteria for diabetes."
            }
        ],
        "memory": [
            {"date": "2023-09-01", "note": "Previous check-up showed borderline glucose levels."}
        ]
    }
}


def run_test_case(case_name: str, case_data: dict):
    """Run a single test case and return agent response + metrics."""
    # Initialize HF agent (replace with your chosen model)
    agent = DiabetesAgentHF(model_name="stabilityai/stablelm-tuned-alpha-7b")  # lightweight HF model

    response = agent.analyze(
        patient_data=case_data["patient_data"],
        classifier_output=case_data["classifier_output"],
        retrieved_chunks=case_data["retrieved_chunks"],
        memory=case_data.get("memory")
    )

    metrics = agent.evaluate_response(response)

    print(f"\n=== {case_name} ===")
    print("Conclusion:", response.conclusion)
    print("Reasoning:", "\n - ".join(response.reasoning))
    print("Next Steps:", "\n - ".join(response.next_steps))
    print("Sources:", [s.dict() for s in response.sources])
    print("Metrics:", metrics)

    return response, metrics


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for case_name, case_data in TEST_CASES.items():
        response, metrics = run_test_case(case_name, case_data)

        # Save structured JSON output for inspection
        output_file = results_dir / f"{case_name}_output.json"
        with open(output_file, "w") as f:
            json.dump(response.dict(), f, indent=2)

        print(f"\nSaved results for {case_name} to {output_file}")


if __name__ == "__main__":
    main()
