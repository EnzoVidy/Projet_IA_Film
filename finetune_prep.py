import pandas as pd
import json
from config import CSV_MOVIES, FINETUNE_OUTPUT

def prepare_mistral_finetuning_data(csv_path=None, output_file=None):
    if csv_path is None:
        csv_path = str(CSV_MOVIES)
    if output_file is None:
        output_file = str(FINETUNE_OUTPUT)
    try:
        df = pd.read_csv(csv_path)
        df = df[['original_title', 'overview']].dropna()
        with open(output_file, 'w', encoding='utf-8') as f:
            for index, row in df.iterrows():
                # format a envoyer pour Mistral
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