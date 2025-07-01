# inside process.py
from evaluateFunction import main as run_eval
from llmFilter import main as run_filter

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: ./process <filename.docx>")
        sys.exit(1)

    docx_file = sys.argv[1]
    print(f"Processing file: {docx_file}")

    run_eval(docx_file)
    # run_filter(docx_file)
if __name__ == "__main__":
    main()