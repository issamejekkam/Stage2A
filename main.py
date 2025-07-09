from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import urllib.parse

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
    

    raw_param = data.param
    param = urllib.parse.unquote_plus(raw_param)
    print(f"🔹 Received param: {param}")
    

    try:
        # Run evaluateFunction.py
        result1 = subprocess.run(
            ["python3", "evaluateFunction.py", param],
            capture_output=True,
            text=True
        )
        print("✅ evaluateFunction.py stdout:\n", result1.stdout)
        if result1.stderr:
            print("❌ evaluateFunction.py stderr:\n", result1.stderr)

        # Run ComparaisonLexicale.py
        result2 = subprocess.run(
            ["python3", "ComparaisonLexicale.py", param],
            capture_output=True,
            text=True
        )
        print("✅ ComparaisonLexicale.py stdout:\n", result2.stdout)
        if result2.stderr:
            print("❌ ComparaisonLexicale.py stderr:\n", result2.stderr)

        return {
            "message": f"Command executed with param: {param}",
        }

    except Exception as e:
        print("🔥 Exception occurred:", e)
        return {"error": str(e)}