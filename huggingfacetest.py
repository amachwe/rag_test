import requests
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
#API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1"

text = "Yes, you can choose to do this under 1) SELECT A CASE. Please note that your case needs to be smaller than 600mm x 300mm x 630mm and weigh less than 24kg."
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response.headers)
    return response.json()[0]['generated_text']
data = query({"inputs": f"given the text: {text}. Can I bring my own case?"})
#"options":{"use_cache": False}
print(data)