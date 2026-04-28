import requests

API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"

def generate_report(predicted_class, confidence, articles, hf_token):
    context = "\n".join([
        f"- {a['title']} ({a['year']}): {a['abstract']}" 
        for a in articles
    ])
    input_text = f"Diagnosis: {predicted_class} ({confidence:.1%}). {context[:400]}. Treatment: surgical resection and radiotherapy recommended."
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, json={"inputs": input_text})
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    return "Report generation failed"

if __name__ == "__main__":
    print("LLM Report module ready")
