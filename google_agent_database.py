import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import spacy
import locationtagger
import csv
# from transformers import pipeline
"""Backward-compatible entry point for generating the travel database."""

from travel_pipeline import DEFAULT_OUTPUT_CSV, DEFAULT_SURVEY_CSV, generate_travel_database


def main() -> None:
    generate_travel_database(DEFAULT_SURVEY_CSV, DEFAULT_OUTPUT_CSV)


if __name__ == "__main__":
    main()