import os
import openai
import logging
from flask import Flask, request, jsonify

# Configure structured logging to be visible in Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# Configure the OpenAI client.
# The API key is read securely from the OPENAI_API_KEY environment variable.
try:
    client = openai.OpenAI()
    logging.info("OpenAI client initialized successfully.")
except openai.OpenAIError as e:
    # This helps diagnose if the API key is missing during startup.
    logging.error(f"Error initializing OpenAI client: {e}", exc_info=True)
    logging.warning("Chat endpoint will be disabled.")
    client = None

@app.route("/")
def index():
    """A simple welcome message for the root endpoint."""
    return jsonify({"message": "Welcome to the FX AI Pipeline! Use the /chat endpoint to interact."})

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
        # Log the detailed error for debugging, but return a generic message to the user.
        logging.error(f"OpenAI API error occurred: {e}", exc_info=True)
        return jsonify({"error": "An error occurred while communicating with the OpenAI API."}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An unexpected server error occurred."}), 500