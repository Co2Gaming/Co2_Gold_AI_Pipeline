import os
import openai
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure the OpenAI client.
# The API key is read securely from the OPENAI_API_KEY environment variable.
try:
    client = openai.OpenAI()
    print("OpenAI client initialized successfully.")
except openai.OpenAIError as e:
    # This helps diagnose if the API key is missing during startup.
    print(f"Error initializing OpenAI client: {e}")
    print("Chat endpoint will be disabled.")
    client = None

@app.route("/healthz")
def health_check():
    """Health check endpoint for Render to ensure the service is running."""
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat_with_gpt():
    """Accepts a prompt and returns a response from OpenAI's chat model."""
    if not client:
        return jsonify({"error": "OpenAI client not initialized. Check API key."}), 503

    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Please include a JSON body with a `prompt` field"}), 400

    try:
        # Using the modern OpenAI SDK for chat completions
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        reply = resp.choices[0].message.content
        return jsonify({"reply": reply})
    except openai.APIError as e:
        return jsonify({"error": f"OpenAI API error: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500