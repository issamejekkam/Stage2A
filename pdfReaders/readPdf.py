from PyPDF2 import PdfReader

def read_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {e}")
        return ""
    
def contain(text, keyword):
    return keyword.lower() in text.lower()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python readPdf.py <pdf_file_path>")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    content = read_pdf(pdf_file_path)
    if content:
        
        print(content)
    else:
        print("No content extracted from the PDF.")
        