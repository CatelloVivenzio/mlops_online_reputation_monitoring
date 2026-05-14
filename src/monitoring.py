import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from evidently import Report
from evidently.presets import DataDriftPreset
import requests

# Carica le variabili d'ambiente dal file .env locale
load_dotenv()

def initialize_reference_dataset():

    """
    Scarica un campione dal dataset originale 'tweet_eval' usato per 
    addestrare il modello RoBERTa e crea il file di 
    baseline 'src/reference.csv' via HTTP.

    """

    REFERENCE_FILE = "src/reference.csv"
    
    # Se il file esiste non fa nulla
    if os.path.exists(REFERENCE_FILE):
        return
        
    try:
        print("[MONITORING] Dynamically generate the reference file from the tweet_eval dataset.")
        
        # Link HTTP API ufficiale di Hugging Face 
        # per lo split di validation (sentiment)
        url = "https://datasets-server.huggingface.co/rows?dataset=cardiffnlp/tweet_eval&config=sentiment&split=validation&offset=100&length=100"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Solleva un errore in caso di codice HTTP fallimentare
        data_json = response.json()
        
        
        # Estrae i record
        rows_data = [row['row'] for row in data_json['rows']]
        df = pd.json_normalize(rows_data)
        
        # Mappa le etichette numeriche (0,1,2) nelle stringhe 
        # (negative, neutral, positive)
        # come quelle restituite dal modello
        label_mapping = {0: "negative", 1: "neutral", 2: "positive"}
        df["label"] = df["label"].map(label_mapping)
        
        # Verifica che la cartella di destinazione esiste
        os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
        
        # Salva i campi utili per Evidently AI senza l'indice
        df[["text", "label"]].to_csv(REFERENCE_FILE, index=False)
        print(f"[MONITORING SUCCESS] Created baseline file: {REFERENCE_FILE}")
        
    except Exception as e:
        print(f"[MONITORING ERROR] Unable to download tweet_eval via HTTP: {e}")

def get_current_data_from_supabase() -> pd.DataFrame:

    """
    Recupera i dati storici delle predizioni salvate 
    nel database centrale Supabase.

    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("[MONITORING CRITICAL] SUPABASE_URL or SUPABASE_KEY missing in .env file.")
        
    print("[MONITORING] Extracting current data from Supabase")
    
    # Inizializza il client Supabase
    supabase: Client = create_client(supabase_url, supabase_key)
    
    # Esegue la query sulla tabella
    # Estrae il testo e l'etichetta predetta
    response = supabase.table("feedback_logs").select("text, label").execute()
    
    if not response.data:
        print("[MONITORING WARNING] No data found on Supabase. Returning an empty set.")
        return pd.DataFrame(columns=["text", "label"])
        
    return pd.DataFrame(response.data)

def generate_drift_report():

    """
    Controlla la presenza del file di baseline, 
    carica i dati correnti dal DB
    e calcola il Data Drift Report usando Evidently.

    """

    # Assicura che il file di baseline esista prima del calcolo
    initialize_reference_dataset()
    
    REFERENCE_FILE = "src/reference.csv"
    
    # Verifica se procedere solo se il file esiste
    if not os.path.exists(REFERENCE_FILE):
        print("[MONITORING ERROR] Unable to calculate drift: reference.csv file missing.")
        return
        
    try:
        # Carica i due dataset quel dal file e quello da supabase
        reference_df = pd.read_csv(REFERENCE_FILE)
        current_df = get_current_data_from_supabase()
        
        if current_df.empty:
            print("[MONITORING SKIP] The current dataset is empty. Unable to calculate data drift.")
            return
            
        print("[MONITORING] Start data drift report calculation.")
        
        # Configurazione del Report per le API di Evidently
        text_report = Report(metrics=[DataDriftPreset(columns=["text"])])
        
        # Esegue il calcolo del drift
        report_result = text_report.run(reference_data=reference_df, current_data=current_df)
        
        # Salva l'output in formato HTML nella cartella static
        os.makedirs("static", exist_ok=True)
        output_html = "static/drift_report.html"
        report_result.save_html(output_html)
        
        print(f"[MONITORING SUCCESS] Data Drift Report successfully saved to: '{output_html}'")
        
    except Exception as e:
        print(f"[MONITORING CRASH] Critical error while processing report: {e}")

if __name__ == "__main__":
    # Consente l'esecuzione diretta del file test
    generate_drift_report()
