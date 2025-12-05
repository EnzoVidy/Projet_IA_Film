import pandas as pd
import json
from config import CSV_MOVIES, FINETUNE_OUTPUT

def prepare_mistral_finetuning_data(csv_path=None, output_file=None):
    """Prépare les données de fine-tuning pour Mistral."""
    if csv_path is None:
        csv_path = str(CSV_MOVIES)
    if output_file is None:
        output_file = str(FINETUNE_OUTPUT)
    """
    Convertit le CSV en format JSONL compatible avec le fine-tuning Mistral.
    On entraîne le modèle à générer un synopsis à partir d'un titre.
    """
    try:
        df = pd.read_csv(csv_path)
        # On filtre pour avoir des données propres
        df = df[['original_title', 'overview']].dropna()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                # Format Chat pour Mistral
                entry = {
                    "messages": [
                        {"role": "user", "content": f"Génère un synopsis pour le film : {row['original_title']}"},
                        {"role": "assistant", "content": row['overview']}
                    ]
                }
                json.dump(entry, f)
                f.write('\n')
        
        print(f"Fichier {output_file} généré avec {len(df)} exemples.")
        print("Étape suivante : Uploader ce fichier sur la plateforme Mistral AI pour lancer le Fine-Tuning.")
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    prepare_mistral_finetuning_data()