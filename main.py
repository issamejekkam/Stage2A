from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import urllib.parse
from typing import List
import json
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

Elements = []
@app.post("/submit")
async def receive_data(blocs: list[InputBloc]):
    filenames = []
    posteids = []
    userids = []
    fonctionPostes = []
    lexicales = []

    for data in blocs:
        print(f"📦 Reçu depuis VB.NET :")
        print(f" - filename: {data.filename}")
        print(f" - posteid: {data.posteid}")
        print(f" - userid: {data.userid}")
        print(f" - fonction/poste: {data.fonctionPoste}")
        print(f" - lexicale: {data.lexicale}")
        Elements.append(data)
        filenames.append(data.filename)
        posteids.append(data.posteid)
        userids.append(data.userid)
        fonctionPostes.append(data.fonctionPoste)
        lexicales.append(data.lexicale)
    print("📦 Liste des éléments reçus :", Elements)

    try:
        # Préparer les arguments sous forme de chaînes JSON pour Windows
        filenames_arg = json.dumps(filenames)
        posteids_arg = json.dumps(posteids)
        userids_arg = json.dumps(userids)
        fonctionPostes_arg = json.dumps(fonctionPostes)
        lexicales_arg = json.dumps(lexicales)

        result1 = subprocess.run(
            [
                "python", "evaluateFunction.py",
                filenames_arg, posteids_arg, userids_arg, fonctionPostes_arg, lexicales_arg
            ],
            capture_output=True,
            text=True
        )
        print(f"Command executed with params: {filenames_arg}, {posteids_arg}, {userids_arg}, {fonctionPostes_arg}, {lexicales_arg}")
        print("✅ evaluateFunction.py stdout:\n", result1.stdout)
        if result1.stderr:
            with open("log.txt", "w", encoding="utf-8") as file:
                file.write(result1.stderr)
            print("❌ evaluateFunction.py stderr written in log.txt\n")

        return {
            "message": f"Command executed with params: {filenames_arg}, {posteids_arg}, {userids_arg}, {fonctionPostes_arg}, {lexicales_arg}",
        }
    except Exception as e:
        print("🔥 Erreur lors de l'exécution:", e)
        return {"error": str(e)}
