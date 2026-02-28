from loader import load_all_pdfs
import os
from chunker import chunk_text


DATA_PATH = "data/Contrat_apprentissage_SIM_P27.pdf"

def main():
    # Vérification de l'existence du chemin
    if not os.path.exists(DATA_PATH):
        print(f"Erreur : Le chemin {DATA_PATH} n'existe pas.")
        return
    
    try:
        text = load_all_pdfs(DATA_PATH)
        print(f"Longueur texte : {len(text)}")
        print("\nAperçu :\n")
        print(text[:1000])
    except Exception as e:
        print(f"Erreur lors du traitement des PDFs : {e}")

if __name__ == "__main__":
    main()
