# fetch_pubmed.py
import requests
import json
import time
import xml.etree.ElementTree as ET

SEARCH_TERMS = [
    "diabetes treatment", "hypertension drugs", "cancer chemotherapy",
    "asthma symptoms", "heart disease medication", "tuberculosis treatment",
    "malaria drugs", "alzheimer treatment", "depression medication",
    "covid symptoms treatment"
]

def search_pubmed(term, max_results=1500):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    res = requests.get(url, params={
        "db": "pubmed", "term": term,
        "retmax": max_results, "retmode": "json"
    })
    return res.json()["esearchresult"]["idlist"]

def fetch_abstracts(pmids):
    abstracts = []
    for i in range(0, len(pmids), 100):
        batch = pmids[i:i+100]
        res = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml"
            }
        )
        try:
            root = ET.fromstring(res.text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    pmid = article.findtext(".//PMID", "")
                    title = article.findtext(".//ArticleTitle", "")
                    # Abstract can be multiple text blocks
                    abstract_parts = article.findall(".//AbstractText")
                    abstract_text = " ".join([
                        (p.text or "") for p in abstract_parts
                    ]).strip()
                    if abstract_text and len(abstract_text) > 100:
                        abstracts.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract_text
                        })
                except Exception:
                    continue
        except ET.ParseError:
            print(f"  ⚠️ XML parse error on batch {i//100 + 1}, skipping")
        print(f"  Batch {i//100 + 1} done — total: {len(abstracts)}")
        time.sleep(0.4)
    return abstracts

all_abstracts = []
seen_pmids = set()

for term in SEARCH_TERMS:
    print(f"\n🔍 Searching: {term}")
    pmids = search_pubmed(term, max_results=1500)
    new_pmids = [p for p in pmids if p not in seen_pmids]
    seen_pmids.update(new_pmids)
    print(f"  Found {len(new_pmids)} new PMIDs")
    abstracts = fetch_abstracts(new_pmids)
    all_abstracts.extend(abstracts)
    print(f"  Total abstracts: {len(all_abstracts)}")

with open("pubmed_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(all_abstracts, f, indent=2)

total_tokens = sum(len(a["abstract"].split()) * 1.3 for a in all_abstracts)
print(f"\n✅ Done! {len(all_abstracts)} abstracts saved")
print(f"📊 Estimated tokens: {int(total_tokens):,}")