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
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import requests
# from bs4 import BeautifulSoup

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.downloader.download('maxent_ne_chunker_tab')
nltk.downloader.download('words')
nltk.downloader.download('treebank')
nltk.downloader.download('maxent_treebank_pos_tagger')
nltk.downloader.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')


#############################################################

# extracting prompt from csv file

df= pd.read_csv("planmorocco.csv")
df['Start Date']=pd.to_datetime(df['Start Date'],dayfirst=True)
df['End Date ']=pd.to_datetime(df['End Date '],dayfirst=True)
destination= df['Country'][0]
interests=df['Activities'][0]
companions=df['Companions'][0]
budget=df['Budget'][0]
days= (df['End Date '][0]-df['Start Date'][0]).days
city=df['City '][0]

prompt = f"Create a detailed trip plan for {days} days in {destination} ({city}) for {companions} interested in {interests} within a {budget} budget. Include activities for morning, afternoon, and night"

###############################################################

# preparing google api 

def initialize_google_gen_ai():
    GOOGLE_API_KEY = "AIzaSyBTBsVDosOieUh5QDTlITgVZ3qnZPumizY"
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

chat = initialize_google_gen_ai()

response = chat.invoke(prompt).content
# print("Generated Text:", response)

#################################################################

# Save the response to a text file
file_path = 'plan.txt'

with open(file_path, "w") as file:

    file.write(response)

file_path='plan.txt'

#################################################################

# text cleaning 



def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,?!:;\'"-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stop_words(tokens):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Read content from the saved text file
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    content = file.read()

cleaned_plan = clean_text(content)
tokens = tokenize_text(cleaned_plan)
filtered_tokens = remove_stop_words(tokens)

# print("Cleaned Plan:", cleaned_plan)

###########################################################

#extracing activites from the plan (spacy model) by ner

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

def extract_activities(text):
    activities = []

    # Preprocess text: Convert to lowercase and split into sentences
    text = text.lower()
    doc = nlp(text)

    # Extract activity-related sentences
    for sent in doc.sents:
        sentence = sent.text
        if any(keyword in sentence for keyword in ['visit', 'explore', 'tour', 'attend', 'enjoy', 'discover', 'ride', 'taste', 'dine','experience','tour', 'visit', 'explore', 'dine', 'taste', 'enjoy', 'take a tour', 'go to', 'attend', 'ride', 'drink', 'settle','see','experience','try','enjoy','do', 'hike', 'walk', 'shop', 'relax', 'swim', 'snorkel', 'scuba dive', 'kayak', 'sail', 'fish', 'camp', 'picnic', 'photograph', 'birdwatch', 'ski', 'snowboard', 'surf', 'windsurf', 'kite surf', 'paraglide', 'bungee jump', 'skydive','private','check in','check out','stay','book','reserve', 'Guided','lunch','dinner','breakfast','brunch']):
            # Extract verbs and noun phrases
            verb_phrases = [token.lemma_ for token in sent if token.pos_ == 'VERB']
            noun_phrases = [chunk.text for chunk in sent.noun_chunks if chunk.root.pos_ == 'NOUN']

            # Form activity description
            if verb_phrases and noun_phrases:
                activity_description = ' '.join(verb_phrases + noun_phrases)
                activities.append(activity_description.strip())

    return activities


def load_text(file_path):

    with open(file_path, 'r') as file:
        text = file.read()
    return text
content = load_text(file_path)
# Extract activities

activities = extract_activities(content)



def clean_text(text):
    # Remove special characters and asterisks
    cleaned_text = re.sub(r"\d{1,2}:\d{2}\s?(am|pm)?", "", text,flags=re.IGNORECASE)
    cleaned_text = re.sub(r'[*]', '', cleaned_text)
    cleaned_text = re.sub(r'[\*\*]', '', cleaned_text)
    cleaned_text = re.sub(r'[^\w\s,.]', '', cleaned_text)
    cleaned_text = re.sub(r"\d+", "", cleaned_text)
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

cleaned_activities = [clean_text(activity) for activity in activities]
# print("Cleaned Activities:", cleaned_activities)

###############################################################


#extracting location from CLEANED PLAN 


place_entity = locationtagger.find_locations(text = cleaned_plan)

# print(place_entity)
# print("The countries in text : ")
# print(place_entity.countries)

# # getting all states
# print("The states in text : ")
# print(place_entity.regions)

# # getting all cities
# print("The cities in text : ")
# print(place_entity.cities)
# print("All other entities in text : ")
# print(place_entity.other)


######################################################

#FULL FUNCTION TO EXTRACT BOTH LOCATION AND ACTIVITY 

#IGNORE wrong locations in place_entity.other
ACTIVITY_STOPWORDS = {
    "day", "afternoon", "evening", "morning", 
    "budget", "dinner", "lunch", "breakfast", 
    "brunch", "afternoon", "night", "stay","enjoy","experience","night","Relax","AM","PM"
}


def is_valid_location(entity):
    if any(word.lower() in entity.lower() for word in ACTIVITY_STOPWORDS):
        return False
    return True



#main function 
def extract_activities_and_locations(text):
    sentences = sent_tokenize(text)
    activities_with_locations = []

    activity_keywords = re.compile(
        r'\b(?:visit|explore|tour|attend|enjoy|discover|ride|taste|dine|experience|'
        r'take a tour|go to|drink|settle|see|try|do|'
        r'hike|walk|shop|relax|swim|snorkel|scuba dive|kayak|sail|fish|camp|picnic|'
        r'photograph|birdwatch|ski|snowboard|surf|windsurf|kite surf|paraglide|'
        r'bungee jump|skydive|'
        r'private|check in|check out|stay|book|reserve|guided|'
        r'lunch|dinner|breakfast|brunch)\b',
        re.IGNORECASE
    )

    for sentence in sentences:
        if activity_keywords.search(sentence):
            doc = nlp(sentence)

            # get first verb
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            if not verbs:
                continue
            main_verb = verbs[0]

            # filter noun phrases
            noun_chunks = []
            for chunk in doc.noun_chunks:
                phrase = clean_text(chunk.text)
                if phrase and phrase.lower() not in ACTIVITY_STOPWORDS:
                    noun_chunks.append(phrase)

            if noun_chunks:
                # join noun phrases with "and" instead of commas
                activity_description = f"{main_verb} " + " and ".join(noun_chunks)
                activity_description = clean_text(activity_description)

                # locations
                place_entity = locationtagger.find_locations(text=sentence)

                cities = [loc for loc in place_entity.cities if is_valid_location(loc)]
                other = [loc for loc in place_entity.other if is_valid_location(loc)]

                activities_with_locations.append({
                    "country": destination,
                    "budget": budget,
                    "activity": activity_description,
                    "regions": place_entity.regions,
                    "cities": cities,
                    "other": other
                })

    return activities_with_locations

activities_with_locations = extract_activities_and_locations(cleaned_plan)


#printing the final results 

# for activity in activities_with_locations:
#     print(f"Country: {activity['country']}")  # Corrected print statement
#     print(f"Activity: {activity['activity']}")
#     if activity['cities']:
#         print(f"Cities: {', '.join(activity['cities'])}")
#     if activity['other']:
#         print(f"Other Locations: {', '.join(activity['other'])}")
#     if 'budget' in activity:
#         print(f"Budget: {activity['budget']}")
#     print()


##########################################################

#creating a csv file to store final data 

def write_activities_to_csv(activities_with_locations, filename):
    # Define the field names for the CSV
    fieldnames = ['country', 'activity', 'location', 'budget']

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data rows
        for activity in activities_with_locations:
            writer.writerow({
                'country': activity['country'],
                'activity': activity['activity'],
                'location': ', '.join(activity['cities']+activity['other']),
                'budget': activity['budget']
            })


write_activities_to_csv(activities_with_locations, 'database.csv')