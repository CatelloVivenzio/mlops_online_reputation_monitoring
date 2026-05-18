import os
from transformers import pipeline
from dotenv import load_dotenv
from supabase import create_client, Client

# --- GESTIONE DINAMICA DEL MODELLO ---
ORIGINAL_MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"
FINE_TUNED_MODEL_PATH = "src/models/fine_tuned_roberta/"
# -----------------------------------------------------------

# Carica le variabili dal file .env
load_dotenv()

class SentimentModel:

    # Inizializzazione globale del client Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key) if supabase_url and supabase_key else None

    """
    Gestisce l'istanza e l'inferenza del modello RoBERTa e il log su Supabase.
    
    """
    
    def __init__(self):

        """
        Inizializza la pipeline di sentiment-analysis per il modello e supabase client.

        """
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        self.supabase = None
        if supabase_url and supabase_key:
            try:
                # Se le chiavi sono presenti ma malformate 
                # create_client potrebbe sollevare un'eccezione
                self.supabase = create_client(supabase_url, supabase_key)
            except Exception as e:
                print(f"[SUPABASE INIT ERROR] Invalid or malformed keys: {e}")
        else:
            print("[SUPABASE INIT WARNING] Credentials not found in the environment.")
        
        # --- MODIFICA MLOPS FASE 3: CARICAMENTO DINAMICO ---
        # Controlla se la pipeline di retraining ha salvato un modello aggiornato localmente
        if os.path.exists(FINE_TUNED_MODEL_PATH) and os.listdir(FINE_TUNED_MODEL_PATH):
            print(f"[MODEL MLOps] Trovato modello riaddestrato localmente! Caricamento da: {FINE_TUNED_MODEL_PATH}")
            self.analyzer = pipeline("sentiment-analysis", model=FINE_TUNED_MODEL_PATH, tokenizer=FINE_TUNED_MODEL_PATH)
        else:
            print(f"[MODEL MLOps] Nessun modello locale trovato. Caricamento modello originale da Hugging Face: {ORIGINAL_MODEL_PATH}")
            self.analyzer = pipeline("sentiment-analysis", model=ORIGINAL_MODEL_PATH)
        # ---------------------------------------------------

    def predict(self, text: str):

        """
        Analizza il testo e restituisce dati pronti per il frontend.
        
        Returns:
            dict: Label, stringa per display e valore numerico per la barra.

        """

        # Nel caso di input vuoto
        if not text or not text.strip():
            return None
            
        result = self.analyzer(text)[0]
        confidence_pct = round(float(result['score']) * 100, 1)

        # Inserimento automatico su supabase
        if self.supabase:
            try:
                self.supabase.table("feedback_logs").insert({
                    "text": text,
                    "label": result['label'].lower(), # Assicura consistenza con gli import di Evidently
                    "score": result['score'] # Rispetta il vincolo NOT NULL della colonna 'score'
                }).execute()
                print(f"[SUPABASE LOG] Record saved successfully for text: {text[:20]}...")
            except Exception as e:
                # Cattura l'errore a log senza far crashare l'interfaccia utente
                print(f"[SUPABASE ERROR] Unable to save prediction to database: {e}")
        else:
            print("[SUPABASE WARNING] Client not initialized: credentials missing from environment.")

        
        return {
            "label": result['label'],
            "confidence_display": f"{confidence_pct}%",
            "confidence_value": confidence_pct
        }

# Istanza globale del modello (Singleton) per ottimizzare 
# l'uso delle risorse.
# Viene importata in api.py per gestire le richieste degli utenti.
#model_instance = SentimentModel()
