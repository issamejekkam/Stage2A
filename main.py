from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FormData(BaseModel):
    param: str

@app.post("/submit")
async def receive_data(data: FormData):
    param = data.param
    print(f"Received in FastAPI: {param}")

    try:
      
        subprocess.run(
            ["python3", "evaluateFunction.py", param],           # Replace with your real command and arguments
            capture_output=True,
            text=True
        )
        subprocess.run(
            ["python3", "ComparaisonLexicale.py", param],           # Replace with your real command and arguments
            capture_output=True,
            text=True
        )

        return {
            "message": f"Command executed with param: {param}",
 
        }

    except Exception as e:
        return {
            "error": str(e)
        }
