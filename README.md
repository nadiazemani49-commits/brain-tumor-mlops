# Plateforme MLOps — Diagnostic Medical Brain Tumor

**PFA 2024/2025 — Etudiante : Nadia Zemani — Encadrant : Pr. Mohamed LAZAAR**

## Description
Plateforme intelligente de diagnostic medical IRM cerebrales.
- Classification : EfficientNet-B0 (4 classes)
- RAG : PubMed + FAISS
- LLM : Mistral 7B
- MLOps : MLflow + Docker + GitHub Actions

## Dataset
Brain Tumor MRI Dataset — 7023 images — 4 classes :
glioma · meningioma · pituitary · notumor

## Structure
brain-tumor-mlops/
├── src/          # Code source
├── notebooks/    # Notebooks Kaggle
├── docker/       # Docker + Compose
└── .github/      # CI/CD GitHub Actions

## Resultats (10% sample)
- val_acc    : 78.7%
- f1_macro   : 0.786
- notumor    : 100% accuracy
