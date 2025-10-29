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

load_dotenv()  # load .env variables

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# --- Memory & Embedding ---
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

    system_prompt = """You are an AI health assistant that provides patient-friendly diabetes analysis in JSON format.
    CRITICAL: Your entire response must be a single, valid JSON object. Do not include any text before or after the JSON.
    Do not use markdown, formatting, or explanatory text. Output only the JSON structure below.

    Required JSON structure:
    {
        "conclusion": "Clear, empathetic summary written directly to the patient (use 'you' language)",
        "understanding": [
            "Simple explanation of test results in patient-friendly terms",
            "Clear breakdown of risk factors without medical jargon"
        ],
        "next_steps": [
            "Specific, actionable steps the patient should take",
            "Clear guidance on when to see a doctor"
        ],
        "lifestyle_changes": [
            "Practical daily habits they can start today",
            "Realistic health goals with clear metrics"
        ],
        "sources": [
            {
                "doc_id": "Source document ID",
                "page": "page number as integer",
                "quote": "Brief quote from medical guidelines",
                "explanation": "How this applies to the patient in simple terms"
            }
        ]
    }

    Style requirements:
    1. Use "you/your" language throughout
    2. Avoid medical jargon - explain terms simply
    3. Make recommendations specific and actionable
    4. Be encouraging and supportive
    5. Keep explanations brief and clear

    IMPORTANT: Return ONLY the JSON object. Do not add any other text, markdown, or formatting."""

    # Format context to encourage JSON response
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
        temperature=0.2,  # Reduced temperature for more consistent formatting
        max_tokens=2048,
        response_format={"type": "json_object"}  # Request JSON response explicitly
    )

    chat_content = completion.choices[0].message.content.strip()
    if not chat_content:
        raise HTTPException(status_code=500, detail="Groq returned empty response")

    try:
        # First try direct JSON parsing
        result = json.loads(chat_content)
    except json.JSONDecodeError:
        try:
            # Extract JSON from the response
            if '```json' in chat_content:
                # Extract from markdown code block
                json_text = chat_content.split('```json')[1].split('```')[0].strip()
            elif '`json' in chat_content:
                # Extract from inline code block
                json_text = chat_content.split('`json')[1].split('`')[0].strip()
            else:
                # Try to find JSON structure by brackets
                start_idx = chat_content.find('{')
                end_idx = chat_content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = chat_content[start_idx:end_idx]
                else:
                    raise ValueError("No JSON structure found in response")
            
            # Try parsing the extracted JSON
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
    required_fields = ['conclusion', 'understanding', 'next_steps', 'lifestyle_changes', 'sources']
    missing_fields = [field for field in required_fields if field not in result]
    if missing_fields:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required fields in response: {', '.join(missing_fields)}"
        )
    
    return result

# --- Build search query ---
def build_search_query(patient_data: dict, classifier_output: dict) -> str:
    risk_level = classifier_output["prediction"]
    return " ".join([
        f"guidelines for {risk_level} diabetes patient",
        f"BMI {patient_data.get('BMI', 'N/A')}",
        f"blood pressure {patient_data.get('BloodPressure', 'N/A')}",
        f"glucose {patient_data.get('Glucose', 'N/A')}"
    ])

# --- Main analysis pipeline ---
@app.post("/analyze")
async def analyze_case(patient_data: PatientData):
    try:
        data_dict = patient_data.dict()

        # 1️⃣ Embedding
        case_summary = f"BMI {data_dict['BMI']}, BP {data_dict['BloodPressure']}, Glucose {data_dict['Glucose']}, Age {data_dict['Age']}"
        embedding = encoder.encode([case_summary])[0]
        embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)

        # 2️⃣ FAISS dimension check
        import faiss
        expected_dim = embedding_np.shape[1]
        if getattr(memory_manager.index, "d", None) != expected_dim:
            memory_manager.index = faiss.IndexFlatL2(expected_dim)
            memory_manager.cases = []

        # 3️⃣ Query memory
        similar_cases = memory_manager.query_similar_cases(embedding_np, k=2, min_similarity=data_dict['min_similarity'])

        # 4️⃣ Classifier output
        classifier_output = predict_risk(data_dict)
        classifier_output['probability'] = float(classifier_output['probability'])  # ensure float for JSON

        # Return cached result if available
        if similar_cases:
            cached_case = similar_cases[0]
            if cached_case.get('synopsis') and cached_case['outcome'] == classifier_output['prediction']:
                syn = cached_case['synopsis']

                # --- Ensure message format matches non-cached responses ---
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

        # 5️⃣ Full analysis
        model_versions = {"classifier": "xgboost-v1.0", "embedding": "all-MiniLM-L6-v2", "llm": "mixtral-8x7b-32768"}
        query = build_search_query(data_dict, classifier_output)
        retrieved_chunks = await search_guidelines(query)

        groq_analysis = await analyze_with_groq(data_dict, classifier_output, retrieved_chunks)

        # --- Safe evidence extraction ---
        sources_list = groq_analysis.get("sources", [])
        evidence = []
        for src in sources_list[:2]:
            if isinstance(src, dict):
                evidence.append({"source": src.get("doc_id", ""), "quote": str(src.get("quote", ""))})
            elif isinstance(src, str):
                evidence.append({"source": "", "quote": str(src)})

        # --- Ensure next_steps is always a list of strings ---
        next_steps = groq_analysis.get("next_steps", [])
        if isinstance(next_steps, str):
            next_steps = [next_steps]
        elif not isinstance(next_steps, list):
            next_steps = []

        # 6️⃣ Create synopsis
        synopsis = {
            "risk_level": classifier_output["prediction"],
            "confidence": classifier_output["probability"],
            "conclusion": groq_analysis.get("conclusion", ""),
            "next_steps": next_steps,
            "evidence": evidence
        }

        # 7️⃣ Store in memory
        case_id = memory_manager.upsert_case(
            case_data=data_dict,
            embedding=embedding_np,
            model_versions=model_versions,
            outcome=classifier_output["prediction"],
            synopsis=synopsis
        )

        # --- Build final message ---
        recommended_actions = "\n".join([f"{i+1}. {step}" for i, step in enumerate(next_steps)])
        sources = ', '.join([src['quote'] for src in evidence if src['quote']])
        message = (
            f"Diabetes Risk Assessment: {synopsis['risk_level'].replace('_', ' ').title()}\n\n"
            f"Summary: {synopsis['conclusion']}\n\n"
            f"Recommended Actions:\n{recommended_actions}"
        )
        if sources:
            message += f"\n\nSources: {sources}"

        return jsonable_encoder({"message": message})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}\n\nTraceback:\n{tb}")
