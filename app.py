from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from travel_pipeline import DEFAULT_OUTPUT_CSV, DEFAULT_SURVEY_CSV, generate_travel_database, write_survey_to_csv

import nltk

import nltk

nltk_resources = [
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab'),
    ('corpora/words', 'words')
]

for path, package in nltk_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Travel Plan Recommendation System")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def parse_multi_value_field(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.replace("\n", ",").split(",") if item.strip()]


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": None,
            "form": {},
        },
    )


@app.post("/generate", response_class=HTMLResponse)
def generate_plan(
    request: Request,
    country: str = Form(...),
    cities: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    interests: str = Form(...),
    companions: str = Form(...),
    budget: str = Form(...),
):
    form_values = {
        "country": country,
        "cities": cities,
        "start_date": start_date,
        "end_date": end_date,
        "interests": interests,
        "companions": companions,
        "budget": budget,
    }

    try:
        city_list = parse_multi_value_field(cities)
        if not city_list:
            raise ValueError("Please provide at least one city.")

        survey = {
            "Country": country.strip(),
            "City": ", ".join(city_list),
            "Start Date": start_date.strip(),
            "End Date": end_date.strip(),
            "Companions": companions.strip(),
            "Activities": interests.strip(),
            "Budget": budget.strip(),
        }
        write_survey_to_csv(survey, DEFAULT_SURVEY_CSV)
        result = generate_travel_database(DEFAULT_SURVEY_CSV, DEFAULT_OUTPUT_CSV)
        output_df = pd.read_csv(DEFAULT_OUTPUT_CSV).fillna("")
        preview_rows = output_df.to_dict(orient="records")[:12]
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(exc),
                "form": form_values,
            },
            status_code=400,
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "survey": result["survey"],
            "prompt": result["prompt"],
            "rows_written": result["rows_written"],
            "preview_rows": preview_rows,
            "output_csv_path": result["output_csv_path"],
            "plan_path": result["plan_path"],
        },
    )


@app.get("/download/database.csv")
def download_database():
    if not DEFAULT_OUTPUT_CSV.exists():
        raise HTTPException(status_code=404, detail="database.csv has not been generated yet.")
    return FileResponse(path=DEFAULT_OUTPUT_CSV, filename="database.csv", media_type="text/csv")


@app.get("/health")
def health():
    return {"status": "ok"}
