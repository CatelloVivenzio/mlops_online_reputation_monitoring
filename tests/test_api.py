import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open
from src.api import app

# Inizializza il TestClient passando l'applicazione FastAPI
client = TestClient(app)

def test_home_page_loading():

    """
    Verifica che l'endpoint GET '/' risponda correttamente (Stato 200)
    e restituisca l'interfaccia HTML del Reputation Monitor.

    """

    response = client.get("/")
    assert response.status_code == 200
    # Verifica che nel testo HTML restituito il titolo sia corretto
    assert "Machine Innovators Inc." in response.text

def test_analyze_sentiment_positive():

    """
    Verifica che l'endpoint POST '/' accetti l'input dal form
    e restituisca l'etichetta di sentiment attesa.

    """
    
    # Simula l'invio del Form HTML usando il parametro 'data' di httpx
    payload = {"text": "This software is very very good."}
    response = client.post("/", data=payload)
    
    assert response.status_code == 200
    # Verifica che il modello riconosca il sentiment 
    # positivo nel testo
    assert "positive" in response.text.lower()

@patch("src.api.generate_drift_report")
@patch("os.path.exists")
def test_report_endpoint_success(mock_exists, mock_generate):

    """
    Verifica che l'endpoint /report generi il monitoraggio 
    e restituisca l'HTML corretto se il file e' presente.

    """

    # Configura i comportamenti fake con Mock
    mock_generate.return_value = None # Evita il calcolo reale di Evidently e Supabase
    mock_exists.return_value = True # Simula che il file 'static/drift_report.html' esista sul disco
    
    # Sostituisce temporaneamente la lettura del file fisico con una stringa HTML finta
    fake_html = "<html><body>Evidently Report Simulation</body></html>"
    
    with patch("builtins.open", mock_open(read_data=fake_html)):
        # Effettua la chiamata di test all'endpoint sincrono
        response = client.get("/report")
        
    # Asserzioni
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert response.text == fake_html
    
    # Verifica che la pipeline di monitoraggio sia stata effettivamente attivata
    mock_generate.assert_called_once()

@patch("src.api.generate_drift_report")
@patch("os.path.exists")
def test_report_endpoint_file_missing(mock_exists, mock_generate):

    """
    Verifica che l'endpoint gestisca correttamente l'errore 404
    nel caso in cui la generazione del file HTML dovesse fallire.

    """

    # Configura il mock per simulare un fallimento 
    # Il file non viene creato
    mock_generate.return_value = None
    mock_exists.return_value = False  
    
    response = client.get("/report")
    
    # Il server deve rispondere con un 404 ed un messaggio di errore testuale
    assert response.status_code == 404
    assert "Error" in response.text