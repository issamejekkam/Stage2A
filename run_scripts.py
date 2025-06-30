import subprocess
file="Test5.docx"
subprocess.run(["python3", "evaluatefunction.py", file])
subprocess.run(["python3", "llmFilter.py"])
