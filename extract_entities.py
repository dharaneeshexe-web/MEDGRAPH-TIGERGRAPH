# extract_entities.py
import json
import re

# Medical dictionaries for matching
DISEASES = [
    "diabetes", "hypertension", "cancer", "tuberculosis", "malaria", "asthma",
    "alzheimer", "depression", "covid", "pneumonia", "hepatitis", "hiv", "aids",
    "arthritis", "obesity", "stroke", "epilepsy", "parkinson", "schizophrenia",
    "leukemia", "lymphoma", "melanoma", "sepsis", "anemia", "fibrosis",
    "carcinoma", "tumor", "infection", "syndrome", "disorder", "failure"
]

DRUGS = [
    "metformin", "insulin", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "amoxicillin", "penicillin", "doxycycline", "ciprofloxacin", "methotrexate",
    "warfarin", "heparin", "statins", "atorvastatin", "lisinopril", "amlodipine",
    "omeprazole", "prednisone", "dexamethasone", "morphine", "codeine", "opioid",
    "antibiotic", "antiviral", "vaccine", "chemotherapy", "immunotherapy", "placebo"
]

SYMPTOMS = [
    "pain", "fever", "fatigue", "nausea", "vomiting", "cough", "dyspnea",
    "headache", "dizziness", "bleeding", "swelling", "inflammation", "rash",
    "diarrhea", "constipation", "insomnia", "anxiety", "dysphagia", "edema",
    "hypertrophy", "atrophy", "necrosis", "fibrosis", "ischemia", "hypoxia"
]

TREATMENTS = [
    "surgery", "chemotherapy", "radiotherapy", "transplant", "dialysis",
    "physiotherapy", "immunotherapy", "phototherapy", "resection", "biopsy",
    "endoscopy", "catheterization", "angioplasty", "bypass", "amputation",
    "rehabilitation", "counseling", "psychotherapy", "intervention", "procedure"
]

def find_matches(text, word_list):
    text_lower = text.lower()
    found = []
    for word in word_list:
        if word in text_lower and word not in found:
            found.append(word)
    return found[:5]  # max 5 per category

def extract_entities(abstract_text):
    return {
        "diseases":   find_matches(abstract_text, DISEASES),
        "drugs":      find_matches(abstract_text, DRUGS),
        "symptoms":   find_matches(abstract_text, SYMPTOMS),
        "treatments": find_matches(abstract_text, TREATMENTS)
    }

# Load abstracts
with open("pubmed_abstracts.json", "r", encoding="utf-8") as f:
    abstracts = json.load(f)

print(f"Processing {len(abstracts[:5000])} abstracts...")
results = []

for i, abstract in enumerate(abstracts[:5000]):
    entities = extract_entities(abstract["abstract"])
    results.append({
        "pmid": abstract["pmid"],
        "title": abstract["title"],
        "abstract": abstract["abstract"],
        "entities": entities
    })

    if (i+1) % 500 == 0:
        print(f"  ✅ {i+1}/5000 — sample: {entities}")

with open("entities.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Done! {len(results)} abstracts processed → entities.json")