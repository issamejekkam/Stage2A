from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import urllib.parse
from typing import List
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class InputBloc(BaseModel):
    filename: str
    posteid: str
    userid: str
    fonctionPoste: str
    lexicale: str

@app.post("/submit")
async def receive_data(blocs: list[InputBloc]):
    for data in blocs:
        print(f"📦 Reçu depuis VB.NET :")
        print(f" - filename: {data.filename}")
        print(f" - posteid: {data.posteid}")
        print(f" - userid: {data.userid}")
        print(f" - fonction/poste: {data.fonctionPoste}")
        print(f" - lexicale: {data.lexicale}")


    try:
        clients_path="C:/Users/FM/Desktop/JekkamWork/Stage2A/clients"
        #result1 = subprocess.run(
        #    ["docker", "run", "-it", "--rm",
        #   "-v", f"{clients_path}:/EvalAutomatique/clients",
        #    "--add-host", "SPARK08:192.168.1.114",
        #    "evalautomatique",
        #    data.filename, data.posteid,data.userid, data.fonctionPoste, data.type,data.lexicale],
        #    capture_output=True,
        #    text=True
        #)

        result1 = subprocess.run(
        ["python3", "evaluateFunction.py", data.filename, data.posteid,data.userid, data.fonctionPoste,data.lexicale],
        capture_output=True,
        text=True)


        print("✅ evaluateFunction.py stdout:\n", result1.stdout)
        if result1.stderr:
            with open("log.txt","w",encoding="utf-8") as file:
                file.write(result1.stderr)
            print("❌ evaluateFunction.py stderr written in log.txt\n")

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
            "message": f"Command executed with param: {data.filename, data.posteid,data.userid, data.fonctionPoste,data.lexicale}",
        }
    except Exception as e:
        print("🔥 Erreur lors de l'exécution:", e)
        return {"error": str(e)}
