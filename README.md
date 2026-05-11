---
title: MachineInnovators - Reputation Monitor
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
---

# MachineInnovators Inc. - Sentiment Analysis & MLOps
**Autore:** Catello Vivenzio
**Modulo:** AI Engineering - Machine Learning in Produzione

## 📋 Descrizione del Progetto
MachineInnovators Inc. è un sistema scalabile per il monitoraggio della reputazione sui social media. Il progetto integra metodologie **MLOps** per automatizzare l'analisi del sentiment, il monitoraggio continuo e il retraining dei modelli.

## 🏗️ Sviluppo del Progetto

### Fase 1: Implementazione Modello & API
- Sviluppo di un backend con **FastAPI** per servire il modello `twitter-roberta-base-sentiment-latest`.
- Creazione di un'interfaccia frontend in **HTML5/JavaScript** per l'interazione utente.
- Containerizzazione dell'intera applicazione tramite **Docker**.

### Fase 2: Pipeline CI/CD
- **Integrazione Continua (CI)**: Test automatici con `pytest` per validare l'input/output delle API.
- **Distribuzione Continua (CD)**: Deploy automatizzato su **Hugging Face Spaces** al superamento dei test su branch `develop` e `main`.

### Fase 3: Monitoraggio & Feedback
- Integrazione di **FastAPI** per l'inferenza del modello.
- Monitoraggio del **Data Drift** e delle metriche di performance tramite **Evidently AI**.
- Pipeline **CI/CD** automatizzata tramite GitHub Actions e Docker.

## 🛠️ Requisiti Tecnici
- **Linguaggio:** Python 3.10+
- **Framework API:** FastAPI
- **Model Provider:** Hugging Face (RoBERTa-base-sentiment)
- **Monitoring:** Evidently AI / Pandas
- **Infrastructure:** Docker & GitHub Actions

## 🚀 Installazione Locale
1. Clonare il repository.
2. Creare un ambiente virtuale: `python -m venv venv`
3. Attivare l'ambiente: `.\venv\Scripts\activate`
4. Installare le dipendenze: `pip install -r requirements.txt`

