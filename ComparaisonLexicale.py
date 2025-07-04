from database import database

database= database("data.db")
database.connect()
resultats=database.read_json("all_matches_of_Test5.docx.json")
print(resultats)




database.close()