"""
Logging utilities for audit trail and debugging.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Constants
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
AUDIT_LOG_PATH = LOGS_DIR / "audit.log"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(AUDIT_LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("diabetes_agent")

class AuditLogger:
    """Logger for tracking all system operations."""
    
    @staticmethod
    def log_request(request_type: str,
                   patient_data: Dict,
                   model_versions: Dict[str, str],
                   classifier_output: Dict = None,
                   retrieved_docs: list = None,
                   shap_values: Dict = None,
                   memory_id: str = None,
                   cached_result: bool = False) -> None:
        """Log a complete request with all relevant data."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_type": request_type,
            "model_versions": model_versions,
            "patient_data": patient_data,  # Should already be anonymized
            "used_cache": cached_result
        }
        
        if classifier_output:
            log_entry["classifier"] = {
                "prediction": classifier_output.get("prediction"),
                "probability": classifier_output.get("probability")
            }
            
        if retrieved_docs:
            log_entry["retrieved_docs"] = [
                {
                    "doc_id": doc["doc_id"],
                    "page_num": doc["page_num"],
                    "snippet": doc["text"][:200] + "..."  # Truncate long texts
                }
                for doc in retrieved_docs[:3]  # Log only top 3 docs
            ]
            
        if shap_values:
            log_entry["shap_summary"] = {
                "top_features": shap_values.get("top_features", []),
                "feature_importance": shap_values.get("importance", {})
            }
            
        if memory_id:
            log_entry["memory_id"] = memory_id
            
        # Write to audit log
        logger.info(json.dumps(log_entry))
        
    @staticmethod
    def log_error(error_type: str, details: Any) -> None:
        """Log error events."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "details": str(details)
        }
        logger.error(json.dumps(error_entry))