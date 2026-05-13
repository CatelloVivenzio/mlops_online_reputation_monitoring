import pytest
from fastapi.testclient import TestClient
# Importa l'app dal modulo api all'interno di src
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