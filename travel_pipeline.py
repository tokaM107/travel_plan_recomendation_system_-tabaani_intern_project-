from __future__ import annotations

import csv
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SURVEY_CSV = BASE_DIR / "for_prompt.csv"
DEFAULT_OUTPUT_CSV = BASE_DIR / "database.csv"
DEFAULT_PLAN_TXT = BASE_DIR / "plan.txt"

SURVEY_COLUMNS = ["Country", "City", "Start Date", "End Date", "Companions", "Activities", "Budget"]
OUTPUT_COLUMNS = ["country", "activity", "location", "budget"]

ACTIVITY_STOPWORDS = {
    "day",
    "afternoon",
    "evening",
    "morning",
    "budget",
    "dinner",
    "lunch",
    "breakfast",
    "brunch",
    "night",
    "stay",
    "enjoy",
    "experience",
    "relax",
    "am",
    "pm",
}

ACTIVITY_KEYWORDS = re.compile(
    r"\b(?:visit|explore|tour|attend|enjoy|discover|ride|taste|dine|experience|"
    r"take a tour|go to|drink|settle|see|try|do|"
    r"hike|walk|shop|relax|swim|snorkel|scuba dive|kayak|sail|fish|camp|picnic|"
    r"photograph|birdwatch|ski|snowboard|surf|windsurf|kite surf|paraglide|"
    r"bungee jump|skydive|private|check in|check out|stay|book|reserve|guided|"
    r"lunch|dinner|breakfast|brunch)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = re.sub(r"\d{1,2}:\d{2}\s?(am|pm)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*]", "", text)
    text = re.sub(r"[\*\*]", "", text)
    text = re.sub(r"[^\w\s,.]", "", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_cities(raw_cities: str) -> list[str]:
    if not raw_cities:
        return []
    parts = re.split(r"[,;\n]+", str(raw_cities))
    return [part.strip() for part in parts if part.strip()]


def format_date_for_csv(value: str) -> str:
    parsed_value = pd.to_datetime(value, dayfirst=True, errors="coerce")
    if pd.isna(parsed_value):
        raise ValueError(f"Invalid date value: {value}")
    return parsed_value.strftime("%d/%m/%Y")


def parse_survey_date(value: str):
    for parse_kwargs in (
        {"format": "%Y-%m-%d"},
        {"format": "%d/%m/%Y"},
        {"dayfirst": True},
    ):
        parsed_value = pd.to_datetime(value, errors="coerce", **parse_kwargs)
        if not pd.isna(parsed_value):
            return parsed_value
    return pd.to_datetime(value, errors="coerce")


def write_survey_to_csv(survey: dict[str, str], output_path: str | Path = DEFAULT_SURVEY_CSV) -> Path:
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SURVEY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "Country": survey["Country"].strip(),
                "City": survey["City"].strip(),
                "Start Date": survey["Start Date"].strip(),
                "End Date": survey["End Date"].strip(),
                "Companions": survey["Companions"].strip(),
                "Activities": survey["Activities"].strip(),
                "Budget": survey["Budget"].strip(),
            }
        )
    return output_path


def load_survey(input_csv_path: str | Path = DEFAULT_SURVEY_CSV) -> dict[str, str]:
    df = pd.read_csv(input_csv_path)
    if df.empty:
        raise ValueError("The survey CSV is empty.")

    df.columns = [str(column).strip() for column in df.columns]
    row = df.iloc[0]

    country = str(row.get("Country", "")).strip()
    cities = parse_cities(str(row.get("City", "")))
    start_date = parse_survey_date(str(row.get("Start Date", "")))
    end_date = parse_survey_date(str(row.get("End Date", "")))

    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("Start Date and End Date must be valid dates.")

    companions = str(row.get("Companions", "")).strip()
    activities = str(row.get("Activities", "")).strip()
    budget = str(row.get("Budget", "")).strip()

    return {
        "Country": country,
        "City": ", ".join(cities) if cities else str(row.get("City", "")).strip(),
        "Cities": cities,
        "Start Date": start_date.strftime("%d/%m/%Y"),
        "End Date": end_date.strftime("%d/%m/%Y"),
        "Companions": companions,
        "Activities": activities,
        "Budget": budget,
        "Days": max((end_date - start_date).days, 1),
    }


def build_prompt_from_survey(survey: dict[str, str]) -> str:
    return (
        f"Create a detailed trip plan for {survey['Days']} days in {survey['Country']} "
        f"({survey['City']}) for {survey['Companions']} interested in {survey['Activities']} "
        f"within a {survey['Budget']} budget. Include activities for morning, afternoon, and night."
    )


def build_fallback_plan(survey: dict[str, str]) -> str:
    cities = survey.get("Cities") or [survey["Country"]]
    days = int(survey.get("Days", 1))
    interest_text = survey.get("Activities", "travel") or "travel"
    companions = survey.get("Companions", "travellers") or "travellers"
    budget = survey.get("Budget", "standard") or "standard"

    lines = [
        f"Trip overview: This {days}-day plan is designed for {companions} who enjoy {interest_text} on a {budget} budget.",
    ]

    for index in range(days):
        city = cities[index % len(cities)]
        lines.append(f"Day {index + 1} in {city}.")
        lines.append(f"Morning: Visit the historic center of {city} and explore local landmarks.")
        lines.append(f"Afternoon: Enjoy a guided tour and taste local food in {city}.")
        lines.append(f"Night: Dine at a recommended restaurant and relax at your hotel in {city}.")

    return "\n".join(lines)


def generate_plan_text(prompt: str, survey: dict[str, str], google_api_key: str | None = None) -> str:
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return build_fallback_plan(survey)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
        response = chat.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception:
        return build_fallback_plan(survey)


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    return spacy.load("en_core_web_sm")


def split_sentences(text: str) -> list[str]:
    try:
        import nltk

        for resource_name, download_name in [
            ("tokenizers/punkt", "punkt"),
            ("tokenizers/punkt_tab", "punkt_tab"),
        ]:
            try:
                nltk.data.find(resource_name)
            except LookupError:
                nltk.download(download_name, quiet=True)

        from nltk.tokenize import sent_tokenize

        return [sentence.strip() for sentence in sent_tokenize(text) if sentence.strip()]
    except Exception:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        normalized_item = item.strip()
        if normalized_item and normalized_item.lower() not in seen:
            seen.add(normalized_item.lower())
            unique_items.append(normalized_item)
    return unique_items


def is_valid_location(entity: str) -> bool:
    entity_lower = entity.lower()
    return not any(word in entity_lower for word in ACTIVITY_STOPWORDS)


def extract_activities_and_locations(text: str, default_country: str, default_budget: str) -> list[dict[str, object]]:
    import locationtagger

    nlp = get_nlp()
    activities_with_locations: list[dict[str, object]] = []

    for sentence in split_sentences(text):
        if not ACTIVITY_KEYWORDS.search(sentence):
            continue

        doc = nlp(sentence)
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

        noun_chunks: list[str] = []
        for chunk in doc.noun_chunks:
            phrase = clean_text(chunk.text)
            if phrase and phrase.lower() not in ACTIVITY_STOPWORDS:
                noun_chunks.append(phrase)

        main_verb = verbs[0] if verbs else ""
        if noun_chunks:
            activity_description = clean_text(f"{main_verb} " + " and ".join(noun_chunks))
        else:
            activity_description = clean_text(sentence)

        place_entity = locationtagger.find_locations(text=sentence)
        cities = unique_preserve_order([location for location in place_entity.cities if is_valid_location(location)])
        other = unique_preserve_order([location for location in place_entity.other if is_valid_location(location)])
        combined_locations = unique_preserve_order(cities + other)

        activities_with_locations.append(
            {
                "country": default_country,
                "budget": default_budget,
                "activity": activity_description,
                "location": ", ".join(combined_locations),
                "cities": cities,
                "other": other,
            }
        )

    return activities_with_locations


def write_activities_to_csv(activities_with_locations: list[dict[str, object]], filename: str | Path) -> Path:
    filename = Path(filename)
    with filename.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for activity in activities_with_locations:
            writer.writerow(
                {
                    "country": str(activity.get("country", "")).strip(),
                    "activity": str(activity.get("activity", "")).strip(),
                    "location": str(activity.get("location", "")).strip(),
                    "budget": str(activity.get("budget", "")).strip(),
                }
            )
    return filename


def read_csv_preview(csv_path: str | Path, limit: int = 12) -> list[dict[str, str]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    return df.head(limit).fillna("").to_dict(orient="records")


def generate_travel_database(
    input_csv_path: str | Path = DEFAULT_SURVEY_CSV,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV,
    plan_path: str | Path = DEFAULT_PLAN_TXT,
    google_api_key: str | None = None,
) -> dict[str, object]:
    survey = load_survey(input_csv_path)
    prompt = build_prompt_from_survey(survey)
    plan_text = generate_plan_text(prompt, survey, google_api_key=google_api_key)

    plan_path = Path(plan_path)
    plan_path.write_text(plan_text, encoding="utf-8")

    cleaned_plan = clean_text(plan_text)
    activities_with_locations = extract_activities_and_locations(
        cleaned_plan,
        default_country=survey["Country"],
        default_budget=survey["Budget"],
    )
    output_csv_path = write_activities_to_csv(activities_with_locations, output_csv_path)

    return {
        "survey": survey,
        "prompt": prompt,
        "plan_text": plan_text,
        "plan_path": str(plan_path),
        "output_csv_path": str(output_csv_path),
        "rows_written": len(activities_with_locations),
        "preview_rows": read_csv_preview(output_csv_path),
    }
