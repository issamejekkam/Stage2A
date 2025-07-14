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
    filename: str
    posteid: str
    userid: str
    fonctionPoste: str
    type: str
    lexicale: str

@app.post("/submit")
async def receive_data(data: FormData):
    print(f"📦 Reçu depuis VB.NET :")
    print(f" - filename: {data.filename}")
    print(f" - posteid: {data.posteid}")
    print(f" - userid: {data.userid}")
    print(f" - fonction/poste: {data.fonctionPoste}")
    print(f" - type: {data.type}")
    print(f" - lexicale: {data.lexicale}")

    try:
        result1 = subprocess.run(
            ["python3", "evaluateFunction.py", data.filename, data.posteid,data.userid, data.fonctionPoste, data.type,data.lexicale],
            capture_output=True,
            text=True
        )
        print("✅ evaluateFunction.py stdout:\n", result1.stdout)
        if result1.stderr:
            print("❌ evaluateFunction.py stderr:\n", result1.stderr)

        # # Run ComparaisonLexicale.py
        # result2 = subprocess.run(
        #     ["python3", "ComparaisonLexicale.py", data.filename, data.posteid,data.userid, data.fonctionPoste, data.type],
        #     capture_output=True,
        #     text=True
        # )
        # print("✅ ComparaisonLexicale.py stdout:\n", result2.stdout)
        # if result2.stderr:
        #     print("❌ ComparaisonLexicale.py stderr:\n", result2.stderr)

        return {
            "message": f"Command executed with param: {data.filename, data.posteid,data.userid, data.fonctionPoste, data.type,data.lexicale}",
        }
    except Exception as e:
        print("🔥 Erreur lors de l'exécution:", e)
        return {"error": str(e)}
