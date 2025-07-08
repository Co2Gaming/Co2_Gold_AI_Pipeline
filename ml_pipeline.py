import os
import openai
import logging
from flask import Flask, request, jsonify

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.error("OPENAI_API_KEY not set!")
    client = None
else:
    openai.api_key = openai_api_key
    try:
        # For the latest SDK
        client = openai.OpenAI()
        logging.info("OpenAI client initialized.")
    except Exception as e:
        logging.exception("Failed to initialize OpenAI client")
        client = None

@app.route("/")
def index():
    return jsonify({"message": "Welcome! POST to /chat"}), 200

@app.route("/healthz")
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    if not client:
        return jsonify({"error": "OpenAI client not initialized"}), 503

    data = request.get_json(force=True, silent=True)
    logging.info(f"Received payload: {data}")

    if not data or "prompt" not in data:
        return jsonify({"error": "Missing `prompt` in JSON body"}), 400

    prompt = data["prompt"]

    try:
        # Switch back to a supported model
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        reply = resp.choices[0].message.content
        logging.info(f"OpenAI replied: {reply}")
        return jsonify({"reply": reply}), 200

    except Exception as e:
        logging.exception("Error while calling OpenAI API")
        # Return the exception message in the JSON (for debugging)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Only enable debug locally; on Render you want production mode.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
