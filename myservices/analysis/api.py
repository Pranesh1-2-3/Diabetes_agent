from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import json
import numpy as np
import asyncio
from dotenv import load_dotenv

load_dotenv()  

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from diabetes_services.memory.utils import CaseMemoryManager
from sentence_transformers import SentenceTransformer

memory_manager = CaseMemoryManager()
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Import services ---
try:
    from myservices.classifier.api.main import predict_risk
    from myservices.rag.api import search_guidelines
except ImportError as e:
    raise ImportError(f"Failed to import services: {e}")

# --- FastAPI app ---
app = FastAPI(title="Diabetes Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Input model ---
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    min_similarity: Optional[float] = 0.975

# --- Groq LLM analysis ---
async def analyze_with_groq(patient_data: dict, classifier_output: dict, retrieved_chunks: list):
    import groq

    if "GROQ_API_KEY" not in os.environ:
        raise HTTPException(status_code=500, detail="Groq API key missing")

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

    system_prompt = """You are an AI health specialist trained to give expert, actionable advice about diabetes based on patient data and clinical guidelines.
    You must act as the medical expert and directly advise the patient — do not refer them to a doctor, clinic, or medical professional. You are giving the steps yourself.

    If the patient's data suggests a likely type of diabetes (Type 1, Type 2, or prediabetes), mention it clearly and explain its implications in the 'understanding' or 'conclusion' sections.

    Output must be a single valid JSON object following exactly this schema:

    {
        "conclusion": "Empathetic summary addressing the patient directly (use 'you' language', and mention the likely type of diabetes if identifiable)",
        "wellness_overview": "Short paragraph summarizing their current health status and long-term wellness focus. Keep tone supportive and human.",
        "understanding": [
            "Explain what the test results mean in simple terms",
            "Briefly describe what this means for their health"
        ],
        "next_steps": [
            "3 concrete steps the patient should take (e.g., exercise, diet, glucose tracking)"
        ],
        "lifestyle_changes": [
            "Daily routines or practical habits the patient can start now"
        ],
        "sources": [
            {
                "title": "Concise source title (e.g., WHO - Self-monitoring guidance)",
                "quote": "Relevant quote from medical guidelines",
                "explanation": "Explain this quote simply"
            }
        ]
    }

    Style rules:
    1. Use 'you' and 'your' language throughout.
    2. Avoid medical jargon and keep tone encouraging.
    3. Always give exactly 3 'next_steps'.
    4. Never recommend medication or mention doctors.
    5. Be concise and supportive.
    6. Respond with ONLY a JSON object — no markdown, no extra text."""





    formatted_context = f"""Analyze the following patient data and provide recommendations in JSON format only:

Patient Data: {json.dumps(patient_data, indent=2)}
Classifier Output: {json.dumps(classifier_output, indent=2)}
Retrieved Guidelines: {json.dumps(retrieved_chunks, indent=2)}

Remember: Respond with ONLY a JSON object. No other text or formatting."""

    completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_context}
        ],
        model="llama-3.1-8b-instant",
        temperature=0.2,  
        max_tokens=2048,
        response_format={"type": "json_object"}  
    )

    chat_content = completion.choices[0].message.content.strip()
    if not chat_content:
        raise HTTPException(status_code=500, detail="Groq returned empty response")

    try:
        result = json.loads(chat_content)
    except json.JSONDecodeError:
        try:
            if '```json' in chat_content:
                json_text = chat_content.split('```json')[1].split('```')[0].strip()
            elif '`json' in chat_content:
                json_text = chat_content.split('`json')[1].split('`')[0].strip()
            else:
                start_idx = chat_content.find('{')
                end_idx = chat_content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = chat_content[start_idx:end_idx]
                else:
                    raise ValueError("No JSON structure found in response")
            
            result = json.loads(json_text)
            
        except Exception as inner_e:
            print(">>> Raw Groq response:", repr(chat_content))
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse Groq response: {str(inner_e)}\nRaw response: {chat_content[:500]}..."
            )
    
    # Validate the result structure
    if not isinstance(result, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid response structure: expected JSON object"
        )
        
    # Ensure all required fields are present
    required_fields = ['conclusion','wellness_overview','understanding', 'next_steps', 'lifestyle_changes', 'sources']
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required fields in response: {', '.join(missing_fields)}"
        )
    
    return result

# --- Build search query ---
def build_search_query(patient_data: dict, classifier_output: dict) -> str:
    """
    Build a context-aware search query based on top SHAP features.
    Falls back to default features if SHAP data not available.
    """
    risk_level = classifier_output.get("prediction", "unknown")
    top_features = classifier_output.get("top_features", [])

    # Use top SHAP features 
    if top_features and all(isinstance(f, str) for f in top_features):
        selected_features = top_features[:3]
    else:
        selected_features = ["BMI", "BloodPressure", "Glucose"]

 
    feature_parts = []
    for feat in selected_features:
        val = patient_data.get(feat)
        if val is not None:
            feature_parts.append(f"{feat.replace('_', ' ').lower()} {val}")
        else:
            feature_parts.append(f"{feat.replace('_', ' ').lower()} N/A")

    query = " ".join([
        f"guidelines for {risk_level} diabetes patient",
        *feature_parts
    ])
    return query


# --- Main analysis pipeline ---
@app.post("/analyze")
async def analyze_case(patient_data: PatientData):
    try:
        data_dict = patient_data.dict()

        case_summary = f"BMI {data_dict['BMI']}, BP {data_dict['BloodPressure']}, Glucose {data_dict['Glucose']}, Age {data_dict['Age']}"
        embedding = encoder.encode([case_summary])[0]
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)

        import faiss
        expected_dim = embedding_np.shape[1]
        if getattr(memory_manager.index, "d", None) != expected_dim:
            memory_manager.index = faiss.IndexFlatL2(expected_dim)
            memory_manager.cases = []

        similar_cases = memory_manager.query_similar_cases(embedding_np, k=2, min_similarity=data_dict['min_similarity'])

        # --- Handle cases with no biomarker input ---
        numeric_fields = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
        if all((data_dict.get(f, 0) == 0 or data_dict.get(f) is None) for f in numeric_fields):
                return jsonable_encoder({
                    "message": (
                        "No biomarker data detected.\n\n"
                        "Please provide numeric values or upload a table image with biomarkers "
                        "so I can analyze your diabetes risk and offer tailored insights."
                    )
                })

        classifier_output = predict_risk(data_dict)
        classifier_output['probability'] = float(classifier_output['probability']) 

        if similar_cases:
            cached_case = similar_cases[0]
            if cached_case.get('synopsis') and cached_case['outcome'] == classifier_output['prediction']:
                syn = cached_case['synopsis']

                recommended_actions = "\n".join([f"{i+1}. {step}" for i, step in enumerate(syn.get("next_steps", []))])
                sources = ', '.join([src['quote'] for src in syn.get("evidence", []) if src.get('quote')])
                message = (
                    f"Diabetes Risk Assessment: {syn['risk_level'].replace('_', ' ').title()}\n\n"
                    f"Summary: {syn.get('conclusion', '')}\n\n"
                    f"Recommended Actions:\n{recommended_actions}"
                )
                if sources:
                    message += f"\n\nSources: {sources}"

                return jsonable_encoder({
                    "message": message
                })

        model_versions = {"classifier": "xgboost-v1.0", "embedding": "all-MiniLM-L6-v2", "llm": "mixtral-8x7b-32768"}
        query = build_search_query(data_dict, classifier_output)
        retrieved_chunks = await search_guidelines(query)

        groq_analysis = await analyze_with_groq(data_dict, classifier_output, retrieved_chunks)

        # --- Safe evidence extraction ---
        sources_list = groq_analysis.get("sources", [])
        evidence = []

        for src in sources_list:
            if isinstance(src, dict):
                source_title = src.get("title") or src.get("doc_id", "Unknown Source")
                quote = str(src.get("quote", "")).strip()
                evidence.append({
                    "source": source_title,
                    "quote": quote
                })
            elif isinstance(src, str):
                evidence.append({"source": "General Guideline", "quote": str(src)})


        next_steps = groq_analysis.get("next_steps", [])
        if isinstance(next_steps, str):
            next_steps = [next_steps]
        elif not isinstance(next_steps, list):
            next_steps = []


        synopsis = {
            "risk_level": classifier_output["prediction"],
            "confidence": classifier_output["probability"],
            "conclusion": groq_analysis.get("conclusion", ""),
            "next_steps": next_steps,
            "evidence": evidence
        }

        # Store in memory
        case_id = memory_manager.upsert_case(
            case_data=data_dict,
            embedding=embedding_np,
            model_versions=model_versions,
            outcome=classifier_output["prediction"],
            synopsis=synopsis
        )

        # --- Build final message ---
        recommended_actions = "\n".join([f"{i+1}. {step}" for i, step in enumerate(next_steps)])
        
        import re

        lines = []
        for src in evidence:
            if src['quote']:
                cleaned_source = re.sub(r'\d+', '', src['source']).strip()
                lines.append(f"{cleaned_source}: {src['quote']}")
        sources = '\n'.join(lines)



        message = (
            f"Diabetes Risk Assessment: {synopsis['risk_level'].replace('_', ' ').title()}\n\n"
            f"Summary: {synopsis['conclusion']}\n\n"
            f"Wellness Overview: {groq_analysis.get('wellness_overview', 'No wellness insights provided.')}\n\n"
            f"Recommended Actions:\n{recommended_actions}"
        )

        if sources:
            message += f"\n\nSources: {sources}"

        return jsonable_encoder({"message": message})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n\nTraceback:\n{tb}")
