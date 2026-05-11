from transformers import pipeline

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:

    """
    Gestisce l'istanza e l'inferenza del modello.
    
    """
    
    def __init__(self):

        """
        Inizializza la pipeline di sentiment-analysis per il modello.

        """
        
        self.analyzer = pipeline("sentiment-analysis", model=MODEL_PATH)

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
        
        return {
            "label": result['label'],
            "confidence_display": f"{confidence_pct}%",
            "confidence_value": confidence_pct
        }

# Istanza globale del modello (Singleton) per ottimizzare 
# l'uso delle risorse.
# Viene importata in api.py per gestire le richieste degli utenti.
model_instance = SentimentModel()
