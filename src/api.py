from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.model import model_instance

app = FastAPI(title="MachineInnovators - Reputation Monitor")

# Monta la cartella static per i file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configura Jinja2 per leggere i file HTML nella cartella static
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


