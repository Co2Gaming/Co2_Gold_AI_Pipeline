import os
import openai
import logging
from flask import Flask, request, jsonify

# ————————————————
# (1) Any other imports you already have, e.g. your ML pipeline stuff
# from your_ml_module import generate_signal
# ————————————————

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)

# ————————————————
# (2) Your existing routes, e.g. the /ai_input.json endpoint
# @app.route("/ai_input.json", methods=["GET"])
# def ai_input(): …
# ————————————————

# ————————————————
# (3) Paste the ChatGPT setup here:
try:
    client = openai.OpenAI()
    logging.info("OpenAI client initialized successfully.")
except openai.OpenAIError as e:
    logging.error(f"Error initializing OpenAI client: {e}", exc_info=True)
    client = None

@app.route("/chat", methods=["POST"])
def chat_with_gpt():
    if not client:
        return jsonify({"error": "OpenAI client not initialized."}), 503

    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Missing `prompt` in JSON body"}), 400

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return jsonify({"reply": resp.choices[0].message.content})
    except Exception as e:
        logging.error(f"/chat error: {e}", exc_info=True)
        return jsonify({"error": "ChatGPT request failed"}), 500
# ————————————————

# (4) Your existing `if __name__ == "__main__":` block (if any)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
