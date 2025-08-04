from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Global assistant instance
assistant = None
assistant_lock = threading.Lock()


@app.route('/initialize', methods=['GET'])
def initialize():
    global assistant
    if assistant is not None:
        return jsonify({"status": "already_initialized"})

    # Import your FinancialAssistant here to avoid circular imports
    from main import FinancialAssistant, get_stocks, debt_snowball_calculator, lebanon_financial_tips

    financial_functions = {
        'stocks': get_stocks,
        'debt_snowball': debt_snowball_calculator,
        'lebanon_advice': lebanon_financial_tips
    }

    with assistant_lock:
        assistant = FinancialAssistant('intents.json', function_mappings=financial_functions)
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Try to load existing model
        try:
            assistant.load_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')
            print("Model loaded successfully")
        except:
            print("Training new model...")
            assistant.train_model(epochs=150, batch_size=32)
            assistant.save_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')

    return jsonify({
        "status": "initialized",
        "device": str(device)
    })


@app.route('/chat', methods=['POST'])
def chat():
    global assistant
    if assistant is None:
        return jsonify({"error": "Assistant not initialized"}), 400

    data = request.json
    message = data.get('message')
    user_id = data.get('user_id', 'default_user')

    if not message:
        return jsonify({"error": "Missing message"}), 400

    response = assistant.process_message(message, user_id)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)