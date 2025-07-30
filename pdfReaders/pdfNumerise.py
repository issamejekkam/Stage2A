# # import aspose.ocr as ocr

# # # Initialize an object of AsposeOcr class
# # api = ocr.AsposeOcr()

# # # Load the scanned PDF file
# # input = ocr.OcrInput(ocr.InputType.PDF)
# # input.add("pdfnum.pdf")

# # # Recognize text with OCR
# # result = api.recognize(input)

# # # Print the output text to the console
# # print(result[0].recognition_text)


# import pdfplumber
# import pytesseract
# from PIL import Image

# def extraire_texte_pdf_scanne(chemin_pdf):
#     """
#     Extrait le texte d'un PDF scanné en utilisant pdfplumber et pytesseract.

#     Args:
#         chemin_pdf (str): Le chemin d'accès au fichier PDF scanné.

#     Returns:
#         str: Le texte extrait du PDF.
#     """
#     texte_extrait = ""
#     try:
#         with pdfplumber.open(chemin_pdf) as pdf:
#             for page in pdf.pages:
#                 # Extraire l'image de chaque page (si elle existe)
#                 image = page.to_image()
#                 if image:
#                     # Convertir l'image en PIL Image
#                     img = image.original
#                     # Utiliser pytesseract pour reconnaître le texte de l'image
#                     texte_page = pytesseract.image_to_string(img)
#                     texte_extrait += texte_page + "\n"
#     except Exception as e:
#         print(f"Erreur lors de l'extraction du texte: {e}")
#         return None
#     return texte_extrait

# # Exemple d'utilisation
# chemin_pdf_a_lire = "pdfnum.pdf"  # Remplacez par le chemin de votre PDF
# texte = extraire_texte_pdf_scanne(chemin_pdf_a_lire)
# print(texte)




from pdf2image import convert_from_path
import pytesseract
from fpdf import FPDF

# Path to the scanned PDF
pdf_path = 'pdfnum.pdf'

# Convert PDF pages to images
images = convert_from_path(pdf_path, dpi=300)

# Extract text from each image page using OCR
full_text = ""
for i, img in enumerate(images):
    text = pytesseract.image_to_string(img, lang='fra')  
    full_text += f"\n--- Page {i+1} ---\n{text}"

# Print or save the output
output_pdf_path = "extrait_texte.pdf"
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

import unicodedata

for line in full_text.split('\n'):
    # Remove or replace problematic Unicode characters
    clean_line = unicodedata.normalize('NFKD', line).encode('ascii', 'ignore').decode('ascii')
    pdf.cell(0, 10, txt=clean_line, ln=1)

pdf.output(output_pdf_path)
print(f"Texte extrait enregistré dans : {output_pdf_path}")
