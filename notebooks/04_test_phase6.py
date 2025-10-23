#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import asyncio
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# --- Memory & Embedding ---
from diabetes_services.memory.utils import CaseMemoryManager
from sentence_transformers import SentenceTransformer

memory_manager = CaseMemoryManager()
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Import your services ---
try:
    from myservices.classifier.api.main import predict_risk
    from myservices.rag.api import search_guidelines
    from myservices.parser.parser import MedicalTableParser
except ImportError as e:
    print("Import Error:", e)
    sys.exit(1)

# --- Groq LLM analysis ---
async def analyze_with_groq(patient_data, classifier_output, retrieved_chunks):
    import groq
    if "GROQ_API_KEY" not in os.environ:
        api_key = input("Enter your Groq API key: ")
        os.environ["GROQ_API_KEY"] = api_key
    groq_client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

    context = f"""
    Patient Data:
    {json.dumps(patient_data, indent=2)}

    Classifier Output:
    {json.dumps(classifier_output, indent=2)}

    Retrieved Medical Guidelines:
    """
    for i, chunk in enumerate(retrieved_chunks, 1):
        context += f"\nChunk {i}:\nDocument: {chunk['doc_id']}\nPage: {chunk.get('page_num', 'N/A')}\nContent: {chunk['text']}\n"

    system_prompt = """You are a medical analysis assistant. Analyze the patient data, classifier output, and medical guidelines provided.
    Your response must be in JSON format with fields: conclusion, reasoning, next_steps, sources."""

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=2048
    )

    try:
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print("Groq parsing failed:", e)
        return None

# --- Utility to make JSON serializable ---
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

# --- Build search query ---
def build_search_query(patient_data: dict, classifier_output: dict) -> str:
    risk_level = classifier_output["prediction"]
    query_parts = [
        f"guidelines for {risk_level} diabetes patient",
        f"BMI {patient_data.get('BMI', 'N/A')}",
        f"blood pressure {patient_data.get('BloodPressure', 'N/A')}",
        f"glucose {patient_data.get('Glucose', 'N/A')}"
    ]
    return " ".join(query_parts)

# --- Main memory-enabled pipeline ---
async def analyze_case_with_memory(patient_data: dict, min_similarity: float = 0.975):
    import faiss

    # 1. Summary embedding
    case_summary = f"BMI {patient_data.get('BMI')}, BP {patient_data.get('BloodPressure')}, Glucose {patient_data.get('Glucose')}, Age {patient_data.get('Age')}"
    embedding = encoder.encode([case_summary])[0]
    embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)

    # 2. Rebuild FAISS index if dimension mismatch
    expected_dim = embedding_np.shape[1]
    if getattr(memory_manager.index, "d", None) != expected_dim:
        memory_manager.index = faiss.IndexFlatL2(expected_dim)
        memory_manager.cases = []

    # 3. Query memory
    similar_cases = memory_manager.query_similar_cases(embedding=embedding_np, k=2, min_similarity=min_similarity)
    classifier_output = predict_risk(patient_data)

    if similar_cases:
        cached_case = similar_cases[0]
        if cached_case.get('synopsis') and cached_case['outcome'] == classifier_output['prediction']:
            return {
                'cached': True,
                'case_id': cached_case['case_id'],
                'similarity': cached_case['similarity'],
                'synopsis': cached_case['synopsis'],
                'outcome': cached_case['outcome']
            }

    # 4. Run full analysis
    model_versions = {"classifier": "xgboost-v1.0", "embedding": "all-MiniLM-L6-v2", "llm": "mixtral-8x7b-32768"}
    query = build_search_query(patient_data, classifier_output)
    retrieved_chunks = await search_guidelines(query)
    groq_analysis = await analyze_with_groq(patient_data, classifier_output, retrieved_chunks)

    synopsis = {
        "risk_level": classifier_output["prediction"],
        "confidence": classifier_output["probability"],
        "conclusion": groq_analysis.get("conclusion", ""),
        "next_steps": groq_analysis.get("next_steps", []),
        "evidence": [{"source": src["doc_id"], "quote": src["quote"]} for src in groq_analysis.get("sources", [])[:2]]
    }

    case_id = memory_manager.upsert_case(
        case_data=patient_data,
        embedding=embedding.astype(np.float32),
        model_versions=model_versions,
        outcome=classifier_output["prediction"],
        synopsis=synopsis
    )

    return {
        'cached': False,
        'case_id': case_id,
        'classifier_output': classifier_output,
        'retrieved_chunks': retrieved_chunks,
        'analysis': groq_analysis,
        'synopsis': synopsis
    }

# --- Entry point ---
async def main():
    # Determine input type: JSON parameters or image
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> OR <patient_data.json>")
        sys.exit(1)

    input_arg = sys.argv[1]

    # 1. Image input → OCR parsing
    if Path(input_arg).exists() and input_arg.lower().endswith((".png", ".jpg", ".jpeg")):
        parser = MedicalTableParser(use_advanced_model=False)
        patient_data = parser.parse_image(input_arg)
    # 2. JSON input
    else:
        try:
            with open(input_arg, 'r') as f:
                patient_data = json.load(f)
        except Exception as e:
            print("Invalid input. Provide a valid image path or JSON file:", e)
            sys.exit(1)

    results = await analyze_case_with_memory(patient_data)
    print(json.dumps(make_json_serializable(results), indent=2))

# --- Run the script ---
if __name__ == "__main__":
    asyncio.run(main())
