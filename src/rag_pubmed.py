import requests
import xml.etree.ElementTree as ET

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

TUMOR_QUERIES = {
    "glioma":      "glioma brain tumor MRI classification treatment",
    "meningioma":  "meningioma brain tumor MRI diagnosis treatment",
    "pituitary":   "pituitary tumor MRI diagnosis treatment",
    "notumor":     "normal brain MRI no tumor findings",
}

def search_pubmed(query: str, max_results: int = 3) -> list:
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
    }
    response = requests.get(PUBMED_SEARCH_URL, params=params)
    data     = response.json()
    ids      = data["esearchresult"]["idlist"]
    return ids

def fetch_abstracts(pubmed_ids: list) -> list:
    if not pubmed_ids:
        return []
    params = {
        "db":      "pubmed",
        "id":      ",".join(pubmed_ids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    response = requests.get(PUBMED_FETCH_URL, params=params)
    root     = ET.fromstring(response.content)
    abstracts = []
    for article in root.findall(".//PubmedArticle"):
        title    = article.findtext(".//ArticleTitle") or "No title"
        abstract = article.findtext(".//AbstractText")  or "No abstract"
        year     = article.findtext(".//PubDate/Year")  or "Unknown"
        abstracts.append({
            "title":    title,
            "abstract": abstract[:500],
            "year":     year,
        })
    return abstracts

def get_medical_context(predicted_class: str, max_results: int = 3) -> str:
    print(f"Recherche PubMed pour : {predicted_class}")
    query    = TUMOR_QUERIES.get(predicted_class, predicted_class)
    ids      = search_pubmed(query, max_results)
    articles = fetch_abstracts(ids)

    if not articles:
        return "Aucun article trouve."

    context = f"Articles medicaux pour {predicted_class}:\n\n"
    for i, art in enumerate(articles, 1):
        context += f"[{i}] {art['title']} ({art['year']})\n"
        context += f"    {art['abstract']}\n\n"
    return context

if __name__ == "__main__":
    # Test rapide
    result = get_medical_context("glioma")
    print(result)
