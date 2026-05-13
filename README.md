---
title: Machine Innovators - Reputation Monitor
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# MachineInnovators Inc. - Sentiment Analysis & MLOps
**Autore:** Catello Vivenzio
**Modulo:** AI Engineering - Machine Learning in Produzione

## Descrizione del Progetto
MachineInnovators Inc. e' un sistema scalabile per il monitoraggio della reputazione sui social media. Il progetto integra metodologie **MLOps** per automatizzare l'analisi del sentiment, il monitoraggio continuo e il retraining dei modelli.

## Sviluppo del Progetto

### Fase 1: Implementazione Modello & API
- Sviluppo di un backend con **FastAPI** per servire il modello `twitter-roberta-base-sentiment-latest`.
- Creazione di un'interfaccia frontend in **HTML5/CSS** per l'interazione utente.
- Containerizzazione dell'intera applicazione tramite **Docker**.

---

### Fase 2: Pipeline CI/CD
- **Integrazione Continua (CI)**: Test automatici con `pytest` per validare l'input/output delle API.
- **Distribuzione Continua (CD)**: Deploy automatizzato su **Hugging Face Spaces** al superamento dei test su branch `develop` e `main`.

---

### Fase 3: Monitoraggio, Sicurezza & CI/CD
Il progetto adotta un sistema di monitoraggio continuo del data drift, l'inferenza software e una gestione sicura delle credenziali centralizzate per il database Supabase.

---

## Architettura della Fase 3
* **FastAPI**: Integrazione del framework per gestire l'inferenza in tempo reale del modello.
* **Evidently AI**: Monitoraggio continuo del Data Drift testuale e delle metriche di performance.
* **GitHub Actions & Docker**: Pipeline CI/CD automatizzata per build, testing e containerizzazione.

---

## Generazione Dinamica del Reference Dataset
Per evitare il caricamento di dati cablati a mano e garantire la massima coerenza statistica, il sistema include una funzione di inizializzazione dinamica (`initialize_reference_dataset` in `src/monitoring.py`): 

* **Bypass HTTP**: Al primo avvio, se il file `src/reference.csv` manca, il codice interroga direttamente l'API REST di Hugging Face secondo le specifiche della [Hugging Face Dataset Viewer](https://huggingface.co/docs/dataset-viewer/index).
* **Data Sourcing**: Scarica un campione di baseline reale dallo split di validation del dataset `cardiffnlp/tweet_eval (sentiment)` usato per addestrare RoBERTa.
* **Drift Analysis**: La suite di Evidently AI confronta i nuovi testi inviati dagli utenti estratti da Supabase con questa baseline per calcolare il report di stabilita'.

---

## Gestione delle Credenziali (.env) e GitHub Secrets
Per consentire agli esaminatori di testare l'applicazione senza esporre chiavi private nel codice pubblico, il progetto adotta una strategia ibrida:

* **Sviluppo Locale**: Le credenziali reali di Supabase devono essere salvate in un file `.env` posizionato nella root principale del progetto. Questo file è inserito nel `.gitignore` e non verrà mai tracciato nei commit.
  ```env
  SUPABASE_URL=supabase.co
  SUPABASE_KEY=your-anon-public-key
  ```
* **Pipeline Automatica (CI/CD)**: Durante l'esecuzione del workflow su GitHub Actions, l'accesso al database è protetto tramite i **GitHub Secrets** (`SUPABASE_URL` e `SUPABASE_KEY`). Prima del lancio dei test con `pytest`, il workflow genera dinamicamente un file `.env` temporaneo criptato all'interno del container Linux, garantendo il superamento dei test di integrazione in sicurezza.

---

## Documentazione Interattiva delle API (FastAPI)
Il server FastAPI espone automaticamente la documentazione interattiva completa di tutte le rotte di inferenza e monitoraggio. Una volta avviato il server in locale (di default su `http://localhost:8000`), gli esaminatori possono testare gli endpoint tramite i seguenti link ufficiali:

* **Swagger UI (Raccomandato)**: [http://localhost:8000/docs](http://localhost:8000/docs) – Consente di testare le chiamate API (es. invio tweet e predizione sentiment) direttamente dal browser tramite un'interfaccia grafica interattiva.
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc) – Offre una documentazione pulita e dettagliata, per la lettura approfondita degli schemi dei dati e dei codici di risposta HTTP.

---

## Requisiti Tecnici
- **Linguaggio:** Python 3.10+
- **Framework API:** FastAPI
- **Model Provider:** Hugging Face (RoBERTa-base-sentiment)
- **Monitoring:** Evidently AI / Pandas
- **Infrastructure:** Docker & GitHub Actions

---

## Installazione Locale
1. Clonare il repository.
2. Creare un ambiente virtuale: `python -m venv venv`
3. Attivare l'ambiente: `.\venv\Scripts\activate`
4. Installare le dipendenze: `pip install -r requirements.txt`

---

## Documentazione di Riferimento
Per approfondimenti sullo sviluppo e sulle tecnologie integrate nel progetto, consultare le documentazioni ufficiali:
- **FastAPI Framework**: [FastAPI Documentation](https://fastapi.tiangolo.com/) per la gestione e l'estensione delle API di inferenza.
- **Hugging Face Transformers**: [Transformers Docs](https://huggingface.co/docs/transformers/index) per l'ottimizzazione del modello.
- **Jinja2 Template Engine**: [Jinja Documentation](https://jinja.palletsprojects.com/en/stable/templates/) per la gestione e il rendering dinamico delle pagine HTML sul frontend.
- **Evidently AI**: [Evidently Docs](https://docs.evidentlyai.com/) per l'implementazione del monitoraggio del Data Drift e delle metriche di performance.
- **GitHub Actions**: [GitHub Actions Quickstart](https://docs.github.com/de/actions/get-started/quickstart) per la configurazione dei flussi CI/CD e l'automazione dei test.


