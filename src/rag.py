import json, pickle, requests
import numpy as np
import xml.etree.ElementTree as ET
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CLASSES = ['glioma', 'meningioma', 'pituitary', 'notumor']
PUBMED_QUERIES = {
    'glioma'     : 'glioma brain tumor MRI classification deep learning',
    'meningioma' : 'meningioma brain MRI diagnosis features',
    'pituitary'  : 'pituitary adenoma tumor MRI diagnosis',
    'notumor'    : 'normal brain MRI findings healthy',
}

def fetch_pubmed(query: str, n: int = 10) -> list:
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'
    ids  = requests.get(f'{base}/esearch.fcgi',
                        params={'db':'pubmed','term':query,'retmax':n,'retmode':'json'},
                        timeout=15).json()['esearchresult']['idlist']
    if not ids: return []
    xml  = requests.get(f'{base}/efetch.fcgi',
                        params={'db':'pubmed','id':','.join(ids),'retmode':'xml'},
                        timeout=20).content
    root = ET.fromstring(xml)
    arts = []
    for a in root.findall('.//PubmedArticle'):
        abstract = a.findtext('.//AbstractText') or ''
        if not abstract: continue
        arts.append({
            'pmid'    : a.findtext('.//PMID') or '',
            'title'   : a.findtext('.//ArticleTitle') or '',
            'abstract': abstract,
            'year'    : a.findtext('.//PubDate/Year') or 'N/A',
        })
    return arts

def build_index(output_dir: str = '.'):
    output_dir = Path(output_dir)
    all_articles = []
    for cls, query in PUBMED_QUERIES.items():
        arts = fetch_pubmed(query, n=10)
        for a in arts: a['cls'] = cls
        all_articles.extend(arts)
        print(f'  {cls:<15}: {len(arts)} articles')
    embedder   = SentenceTransformer('all-MiniLM-L6-v2')
    texts      = [f"Title: {a['title']}\nAbstract: {a['abstract']}" for a in all_articles]
    embeddings = embedder.encode(texts, show_progress_bar=True).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / 'pubmed_faiss.index'))
    with open(output_dir / 'pubmed_articles.pkl', 'wb') as f:
        pickle.dump(all_articles, f)
    print(f'Index FAISS: {index.ntotal} vecteurs')
    return index, all_articles, embedder

def load_index(output_dir: str = '.'):
    output_dir = Path(output_dir)
    index    = faiss.read_index(str(output_dir / 'pubmed_faiss.index'))
    with open(output_dir / 'pubmed_articles.pkl', 'rb') as f:
        articles = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return index, articles, embedder

def retrieve(query: str, index, articles, embedder, top_k: int = 3) -> str:
    q = embedder.encode([query]).astype(np.float32)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    parts = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), 1):
        a = articles[idx]
        parts.append(
            f'[Article {rank} | PMID {a[\"pmid\"]} | {a[\"year\"]}]\n'
            f'Titre: {a[\"title\"]}\n'
            f'Resume: {a[\"abstract\"][:400]}...\n'
            f'Score: {score:.3f}'
        )
    return '\n\n'.join(parts)

if __name__ == '__main__':
    print('Construction index RAG...')
    build_index('.')
