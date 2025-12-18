import os
from mistralai import Mistral
from config import FINETUNE_OUTPUT

def launch_mistral_finetuning():
    """
    1. Upload le fichier JSONL vers Mistral.
    2. Lance le job de fine-tuning.
    Retourne l'ID du job et l'ID du modèle (futur).
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return "Erreur : Clé API manquante.", None

    client = Mistral(api_key=api_key)
    file_path = str(FINETUNE_OUTPUT)

    if not os.path.exists(file_path):
        return f"Erreur : Le fichier {file_path} n'existe pas. Génère-le d'abord.", None

    try:
        # 1. Upload du fichier d'entrainement
        print("📤 Upload du fichier vers Mistral...")
        
        with open(file_path, "rb") as f:
            # On lit tout le contenu du fichier en binaire
            file_bytes = f.read()
            
            training_file = client.files.upload(
                file={
                    "file_name": "mistral_finetune.jsonl",
                    "content": file_bytes,
                },
                purpose="fine-tune"
            )
        
        print(f"✅ Fichier uploadé. ID: {training_file.id}")

        # 2. Lancement du Job
        print("🚀 Lancement du job de fine-tuning...")
        
        created_job = client.fine_tuning.jobs.create(
            model="open-mistral-7b",
            training_files=[{
                "file_id": training_file.id,
                "weight": 1
            }],
            hyperparameters={
                "training_steps": 100,
                "learning_rate": 0.0001
            },
            auto_start=True 
        )

        job_id = created_job.id
        return f"Succès ! Job lancé sous l'ID : {job_id}", job_id

    except Exception as e:
        return f"Une erreur est survenue : {str(e)}", None