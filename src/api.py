from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.model import model_instance
from src.monitoring import generate_drift_report
import os

app = FastAPI(title="MachineInnovators - Reputation Monitor")

# Monta la cartella static per il file HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configura Jinja2 per leggere il file HTML nella cartella static
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    """
    Serve la pagina HTML iniziale.
    
    """
    
    return templates.TemplateResponse(
        request=request, 
        name="home.html", 
        context={}
    )

@app.get("/report", response_class=HTMLResponse)
def view_monitoring_report():

    """
    Genera il report aggiornato attraverso i dati da Supabase
    e restituisce la pagina HTML interattiva di Evidently.

    """

    # Calcola il report aggiornato
    generate_drift_report()
    
    report_path = "static/drift_report.html"
    
    # Se il calcolo fallisce o non ci sono dati, rimanda un avviso
    if not os.path.exists(report_path):
        return HTMLResponse(
            content="<h3>Error: The report was not generated. Please check your server logs.</h3>", 
            status_code=404
        )
    
    print("[API SUCCESS] Report generated successfully. Reading file.")
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    print("[API SUCCESS] Report submission completed.")
    return HTMLResponse(content=html_content)

@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, text: str = Form(...)):

    """
    Riceve il testo dal Form e restituisce i risultati.
    
    """

    prediction = model_instance.predict(text)

    label_lower = prediction["label"].lower() if prediction else ""
    
    render_context = {
        "request": request,
        "text": text,
        "label": prediction["label"] if prediction else None,
        "confidence": prediction["confidence_display"] if prediction else None,
        "conf_val": prediction["confidence_value"] if prediction else 0,
        "sentiment_key": label_lower[:3] if label_lower else ""
    }
    
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context=render_context
    )


