from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import logging
import json
from main import FinancialAssistant, get_stocks, debt_snowball_calculator, lebanon_financial_tips

app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='api.log',
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Initialize assistant in a thread-safe way
assistant_lock = threading.Lock()
assistant = None


def initialize_assistant():
    global assistant
    with assistant_lock:
        if assistant is None:
            try:
                financial_functions = {
                    'stocks': get_stocks,
                    'debt_snowball': debt_snowball_calculator,
                    'lebanon_advice': lebanon_financial_tips
                }
                assistant = FinancialAssistant('intents.json', function_mappings=financial_functions)
                model_files = ['best_model.pth', 'vectorizer.pkl', 'dimensions.json']
                if all(os.path.exists(f) for f in model_files):
                    logging.info("Loading existing financial model...")
                    assistant.load_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')
                else:
                    logging.info("Training new financial model...")
                    assistant.train_model(epochs=150, batch_size=32)
                    assistant.save_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')
                logging.info("Assistant initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize assistant: {str(e)}")
                raise


# Initialize assistant at startup
initialize_assistant()


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or not isinstance(data.get('message'), str) or not data.get('message').strip():
        logging.warning("Invalid or missing message in request")
        return jsonify({
            'success': False,
            'error': 'Message is required and must be a non-empty string'
        }), 400

    message = data.get('message').strip()
    user_id = data.get('user_id', 'default_user')

    try:
        response = assistant.process_message(message, user_id)
        # Check if response contains Chart.js data
        chart_data = None
        if '```chartjs\n' in response:
            try:
                chart_start = response.index('```chartjs')
                chart_end = response.index('\n```', chart_start)
                chart_data = json.loads(response[chart_start:chart_end])
                response = response[:chart_start-10] + response[chart_end+4:]
            except Exception as e:
                logging.warning(f"Failed to parse Chart.js data: {str(e)}")

        logging.info(f"User {user_id} input: {message}, Response: {response}")
        return jsonify({
            'success': True,
            'response': response.strip(),
            'chart': chart_data,
            'user_id': user_id
        })
    except Exception as e:
        logging.error(f"Error processing message '{message}' for user {user_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    status = 'healthy' if assistant is not None else 'unhealthy'
    return jsonify({'status': status, 'assistant_ready': assistant is not None})



if __name__ == '__main__':
    initialize_assistant()
    app.run(port=5000, debug=True)