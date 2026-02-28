from pypdf import PdfReader
from pathlib import Path


def load_pdf(file_path: str) -> str:
    """
    Charge un fichier PDF et retourne son texte brut.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_all_pdfs(folder_path: str) -> str:
    """
    Charge tous les PDF d'un dossier et concatène leur contenu.
    """
    folder = Path(folder_path)
    all_text = ""

    for pdf_file in folder.glob("*.pdf"):
        print(f"Chargement : {pdf_file.name}")
        all_text += load_pdf(str(pdf_file)) + "\n"

    return all_text