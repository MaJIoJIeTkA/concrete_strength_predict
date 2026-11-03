from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import os
import joblib
import numpy as np
from model import ConcreteModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Загружаем модель и скейлер
MODEL_PATH = "model/model.pth"
SCALER_PATH = "model/scaler.pkl"

model = ConcreteModel()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse(
        request, "index.html", {"prediction": None}
    )

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    cement: float = Form(...),
    slag: float = Form(...),
    ash: float = Form(...),
    water: float = Form(...),
    superplasticizer: float = Form(...),
    coarse_agg: float = Form(...),
    fine_agg: float = Form(...),
    age: float = Form(...)
):
    # Формируем сырой массив
    raw_data = np.array([[
        cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age
    ]])
    
    # Масштабируем так же, как при обучении
    if scaler is not None:
        scaled_data = scaler.transform(raw_data)
    else:
        scaled_data = raw_data
        
    input_tensor = torch.FloatTensor(scaled_data)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
        
    return templates.TemplateResponse(
        request, "index.html", {"prediction": round(prediction, 2)}
    )