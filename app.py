import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, render_template

# Initialize the Flask application
app = Flask(__name__)

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token="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")
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, token="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")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sentiment_labels = ["negative", "neutral", "positive"]

def classify_sentence(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    sentiment = sentiment_labels[predicted_class]

    return sentiment

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling the form submission
@app.route('/classify', methods=['POST'])
def classify():
    sentence = request.form['sentence']
    sentiment = classify_sentence(sentence)
    return render_template('result.html', sentence=sentence, sentiment=sentiment)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
