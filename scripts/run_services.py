"""Script to run all API services needed for the diabetes analysis system."""

import subprocess
import sys
from pathlib import Path

def run_services():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define the services to run
    services = [
        {
            "name": "Classifier API",
            "module": "myservices.classifier.api.main:app",
            "port": 8001
        },
        {
            "name": "RAG API",
            "module": "myservices.rag.api:app",
            "port": 8002
        },
        {
            "name": "Parser API",
            "module": "myservices.parser.api.main:app",
            "port": 8000
        },
        {
            "name": "Analysis API",
            "module": "myservices.analysis.api:app",
            "port": 8003
        }
    ]
    
    # Start each service in a new window
    processes = []
    for service in services:
        python_path = f'"{sys.executable}"'  # Quote the Python path
        cmd = [
            python_path,
            "-m",
            "uvicorn",
            service["module"],
            f"--port={service['port']}",
            "--reload"
        ]
        
        # On Windows, use 'start' to open new windows
        if sys.platform == 'win32':
            window_title = f"Python - {service['name']}"
            start_cmd = f'start "[{service["name"]}]" cmd /k {" ".join(cmd)}'
            subprocess.Popen(start_cmd, shell=True, cwd=str(project_root))
        else:
            # On Unix, use terminal emulator
            subprocess.Popen(cmd)
        
        print(f"Started {service['name']} on port {service['port']}")
        print(f"API docs available at: http://localhost:{service['port']}/docs")
        
    print("\nAll services started! Press Ctrl+C in each window to stop the services.")

if __name__ == "__main__":
    run_services()