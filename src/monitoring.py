import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from evidently import Report
from evidently.presets import DataDriftPreset
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import shutil

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
        
        dataset_drift = False
        
        try:
            # Legge il file HTML generato da Evidently per verificare l'esito
            with open(output_html, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Se Evidently rileva un drift sul dataset, inserisce nel codice HTML
            # della dashboard la stringa di stato "Dataset drift detected" o "Drift detected"
            # Verifichiamo la stringa esatta del verdetto statistico di Evidently
            if "dataset drift is detected" in html_content.lower() and "is not detected" not in html_content.lower():
                dataset_drift = True
                print("[MONITORING SUCCESS] Statistical result detected by HTML: Data Drift Present.")
            else:
                dataset_drift = False
                print("[MONITORING SUCCESS] Statistical result detected by HTML: No Data Drift.")
                
        except Exception as html_error:
            print(f"[MONITORING WARNING] Unable to inspect HTML: {html_error}")
            dataset_drift = False
        
        # Se c'e' un drift avvia il retraining automatico
        if dataset_drift:
            trigger_model_retraining(current_df)
        else:
            print("[MONITORING] No significant data drift detected. Retraining is not needed.")

    except Exception as e:
        print(f"[MONITORING CRASH] Critical error while processing report: {e}")

def trigger_model_retraining(current_df: pd.DataFrame):

    """
    Esegue il FINE-TUNING REALE del modello RoBERTa utilizzando
    i nuovi dati estratti storicamente da Supabase.

    """

    print("[RETRAINING] Launch of the Fine-Tuning pipeline on the RoBERTa model.")

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    OUTPUT_DIR = "src/models/fine_tuned_roberta/"
    TEMP_SAVE_DIR = "src/models/tmp_fine_tuned_roberta"

    try:
        # Preparazione dei dati: mappa le label testuali negli ID numerici richiesti dal modello
        id_mapping = {"negative": 0, "neutral": 1, "positive": 2}

        current_df = current_df.dropna(subset=["text", "label"])
        current_df["label"] = current_df["label"].map(id_mapping)
        current_df = current_df.dropna(subset=["label"])
        current_df["label"] = current_df["label"].astype(int)

        if len(current_df) < 5:
            print("[RETRAINING ABORT] Insufficient data on Supabase for fine-tuning (at least 5 samples needed).")
            return

        # Trasforma il DataFrame in Dataset di Hugging Face
        dataset = Dataset.from_pandas(current_df[["text", "label"]])

        # Tokenizzazione
        print("[RETRAINING] Tokenization of the new dataset in progress")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Split Train/Test per monitorare l'addestramento
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42) if len(current_df) >= 10 else {"train": tokenized_dataset, "test": tokenized_dataset}

        # Caricamento modello base
        print(f"[RETRAINING] Loading the base model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

        # Configurazione iperparametri ottimizzati per CPU
        training_args = TrainingArguments(
            output_dir="./tmp_trainer",          
            num_train_epochs=3,                  
            per_device_train_batch_size=2,       
            per_device_eval_batch_size=2,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=5,
            report_to="none",
            eval_strategy="epoch" if len(current_df) >= 10 else "no",
            save_strategy="no",                  
            use_cpu=True                         
        )

        # Inizializzazione Trainer ed esecuzione addestramento
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset.get("test")
        )

        print("[RETRAINING] Start Computing Fine Tuning")
        trainer.train()

        # Salvataggio definitivo del modello ottimizzato
        os.makedirs(TEMP_SAVE_DIR, exist_ok=True)
        
        print("[RETRAINING] Writing new weights to the temporary directory.")
        trainer.save_model(TEMP_SAVE_DIR)
        tokenizer.save_pretrained(TEMP_SAVE_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("[RETRAINING] Synchronization of configuration files and weights.")
        for filename in os.listdir(TEMP_SAVE_DIR):
            src_file = os.path.join(TEMP_SAVE_DIR, filename)
            dst_file = os.path.join(OUTPUT_DIR, filename)
            try:
                # copy2 sovrascrive i file gestendo i flussi binari a basso livello
                shutil.copy2(src_file, dst_file)
            except Exception as copy_error:
                # Se un file specifico (es. il binario pesante) è bloccato da Uvicorn, 
                # crea una versione contrassegnata con timestamp. Al riavvio l'API la legge.
                import time
                timestamp_filename = f"model_{int(time.time())}.safetensors" if "safetensors" in filename else f"patched_{filename}"
                shutil.copy2(src_file, os.path.join(OUTPUT_DIR, timestamp_filename))
        
        # Pulizia della cartella temporanea
        shutil.rmtree(TEMP_SAVE_DIR, ignore_errors=True)

        REFERENCE_FILE = "src/reference.csv"
        current_df.to_csv(REFERENCE_FILE, index=False)
        print(f"[MLOps PIPELINE] Baseline updated with current data in '{REFERENCE_FILE}' to reset the drift calculation.")

        print(f"[RETRAINING SUCCESS] Fine-tuning completed successfully! Model saved in: {OUTPUT_DIR}")

    except Exception as training_error:
        print(f"[RETRAINING ERROR] Critical error during model fine-tuning: {training_error}")



if __name__ == "__main__":
    # Consente l'esecuzione diretta del file test
    generate_drift_report()
