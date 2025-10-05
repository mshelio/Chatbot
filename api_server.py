import os
import json
import random
import re
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from flask import Flask, request, jsonify
import logging
import requests
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Download required NLTK data
nltk.download('punkt', quiet=True)3
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


# Define EnhancedChatbotModel (same as before)
class EnhancedChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


class StatisticsAPI:
    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.auth_token = auth_token
        self.headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

    def get_weekly_stats(self):
        try:
            response = requests.get(
                f"{self.base_url}/stats/weekly",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Statistics API call failed: {str(e)}")
            return None

    def get_monthly_stats(self):
        try:
            response = requests.get(
                f"{self.base_url}/stats/monthly",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Statistics API call failed: {str(e)}")
            return None

    def get_yearly_stats(self):
        try:
            response = requests.get(
                f"{self.base_url}/stats/yearly",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Statistics API call failed: {str(e)}")
            return None

    def get_top_categories(self, period='monthly'):
        try:
            response = requests.get(
                f"{self.base_url}/stats/top-categories",
                headers=self.headers,
                params={'period': period},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logging.error(f"Top categories API call failed: {str(e)}")
            return None


class FinancialAssistant:
    def __init__(self, intents_path, function_mappings=None, stats_api=None):
        self.model = None
        self.intents_path = intents_path
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.synonyms = {
            "debt": ["loan", "balance", "owe", "liability"],
            "save": ["set aside", "reserve", "accumulate"],
            "invest": ["put money", "allocate", "grow wealth"],
            "budget": ["spending plan", "financial plan", "allocation"],
            "emergency": ["crisis", "unexpected", "rainy day"],
            "stats": ["statistics", "analytics", "report", "summary", "overview"],
            "spending": ["expenses", "outgoings", "costs", "expenditure"],
            "income": ["earnings", "revenue", "salary", "wages"]
        }
        self.documents = []
        self.intents = []
        self.intent_responses = {}
        self.function_mappings = function_mappings or {}
        self.stats_api = stats_api
        self.context = defaultdict(str)

    def enhanced_tokenize(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = nltk.word_tokenize(text.lower())
        processed_words = []
        for word in words:
            for base, syn_list in self.synonyms.items():
                if word in syn_list:
                    word = base
                    break
            lemma = self.lemmatizer.lemmatize(word)
            if lemma not in self.stop_words and len(lemma) > 2:
                processed_words.append(lemma)
        return processed_words

    def parse_intents(self):
        try:
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Intents file not found: {self.intents_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in intents file: {e}")
            raise

        all_patterns = []
        intent_labels = []
        for intent in intents_data['intents']:
            if 'tag' not in intent or 'patterns' not in intent or 'responses' not in intent:
                logging.error(f"Invalid intent structure: {intent}")
                raise ValueError("Each intent must have 'tag', 'patterns', and 'responses'")
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intent_responses[tag] = intent['responses']
            for pattern in intent['patterns']:
                processed_text = " ".join(self.enhanced_tokenize(pattern))
                if not processed_text:
                    logging.warning(f"Empty processed text for pattern: {pattern}")
                    continue
                all_patterns.append(processed_text)
                self.documents.append((processed_text, tag))
                intent_labels.append(tag)
        if not all_patterns:
            logging.error("No valid patterns found in intents.json")
            raise ValueError("No valid patterns to train model")
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(all_patterns).toarray()
        self.labels = np.array([self.intents.index(tag) for tag in intent_labels])
        logging.debug(f"Parsed intents: {len(self.intents)} intents, {len(all_patterns)} patterns")
        return X, self.labels

    def prepare_data(self):
        X, y = self.parse_intents()
        return X, y

    def train_model(self, batch_size=32, lr=0.001, epochs=150, patience=5):
        X, y = self.prepare_data()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = EnhancedChatbotModel(X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        best_loss = float('inf')
        patience_counter = 0
        logging.info("Training Financial Advice Model...")
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(loader)
            scheduler.step(epoch_loss)
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
        self.model.load_state_dict(torch.load('best_model.pth'))
        logging.info("Training complete!")

    def save_model(self, model_path, vectorizer_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(vectorizer_path, 'wb') as v_file:
            pickle.dump(self.vectorizer, v_file)
        with open(dimensions_path, 'w') as f:
            json.dump({
                'input_size': len(self.vectorizer.get_feature_names_out()),
                'output_size': len(self.intents),
                'intents': self.intents
            }, f)

    def load_model(self, model_path, vectorizer_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        with open(vectorizer_path, 'rb') as v_file:
            self.vectorizer = pickle.load(v_file)
        self.intents = dimensions['intents']
        self.model = EnhancedChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message, user_id=None, auth_token=None):
        # Update stats API with current user's token
        if auth_token and self.stats_api:
            self.stats_api.auth_token = auth_token
            self.stats_api.headers['Authorization'] = f'Bearer {auth_token}'

        if user_id:
            input_lower = input_message.lower()
            if "lebanon" in input_lower:
                self.context[user_id] = "lebanon"
            elif "debt" in input_lower:
                self.context[user_id] = "debt"
            elif "emergency" in input_lower:
                self.context[user_id] = "emergency"
            elif any(word in input_lower for word in ['stats', 'statistics', 'spending', 'income', 'budget']):
                self.context[user_id] = "statistics"

        processed_input = " ".join(self.enhanced_tokenize(input_message))
        input_vec = self.vectorizer.transform([processed_input]).toarray()
        input_tensor = torch.tensor(input_vec, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()

        predicted_intent = self.intents[pred_idx]

        if confidence < 0.6:
            context = self.context.get(user_id, "")
            if context == "lebanon":
                return "Could you clarify your Lebanon-specific financial question?"
            elif context == "debt":
                return "Could you provide more details about your debt situation?"
            elif context == "statistics":
                return "I'd love to help with your financial stats! Could you specify if you want weekly, monthly, or yearly statistics?"
            return "I'm not sure I understand. Could you rephrase your financial question?"

        func_response = ""
        if predicted_intent in self.function_mappings:
            try:
                func_response = self.function_mappings[predicted_intent](user_id)
                if not isinstance(func_response, str):
                    func_response = str(func_response)
            except Exception as e:
                logging.error(f"Function error: {e}")
                func_response = ""

        response = random.choice(self.intent_responses[predicted_intent])
        context = self.context.get(user_id, "")

        if context == "lebanon":
            response += "\n\nFor Lebanon: Consider dollarizing assets and diversifying banks."
        elif context == "debt":
            response += "\n\nRemember: Attack smallest debts first with 'gazelle intensity'!"
        elif context == "statistics" and self.stats_api:
            # Add relevant statistics based on the conversation
            stats_context = self.get_relevant_stats(input_message)
            if stats_context:
                response += f"\n\n{stats_context}"

        if func_response:
            response += f"\n{func_response}"

        return response

    def get_relevant_stats(self, message):
        """Extract and return relevant statistics based on the message content"""
        message_lower = message.lower()

        if any(word in message_lower for word in ['week', 'weekly']):
            stats = self.stats_api.get_weekly_stats()
            if stats and stats.get('success'):
                data = stats['data']
                return f"Weekly Summary: Income: {data['summary']['total_income']} {data['currency']}, Expenses: {data['summary']['total_expenses']} {data['currency']}"

        elif any(word in message_lower for word in ['month', 'monthly']):
            stats = self.stats_api.get_monthly_stats()
            if stats and stats.get('success'):
                data = stats['data']
                return f"Monthly Summary: Income: {data['summary']['total_income']} {data['currency']}, Expenses: {data['summary']['total_expenses']} {data['currency']}"

        elif any(word in message_lower for word in ['year', 'yearly']):
            stats = self.stats_api.get_yearly_stats()
            if stats and stats.get('success'):
                data = stats['data']
                return f"Yearly Summary: Income: {data['summary']['total_income']} {data['currency']}, Expenses: {data['summary']['total_expenses']} {data['currency']}"

        elif any(word in message_lower for word in ['category', 'categories', 'spending']):
            stats = self.stats_api.get_top_categories()
            if stats and stats.get('success'):
                data = stats['data']
                if data.get('categories'):
                    top_cats = data['categories'][:3]  # Show top 3
                    cat_info = ", ".join([f"{cat['category']} ({cat['percentage']}%)" for cat in top_cats])
                    return f"Top Spending Categories: {cat_info}"

        return None


# Financial functions with statistics integration
def get_weekly_stats(assistant, user_id):
    if assistant.stats_api:
        stats = assistant.stats_api.get_weekly_stats()
        if stats and stats.get('success'):
            data = stats['data']
            summary = data['summary']
            return f"📊 Weekly Financial Summary:\nIncome: {summary['total_income']} {data['currency']}\nExpenses: {summary['total_expenses']} {data['currency']}\nNet Flow: {summary['net_flow']} {data['currency']}\nSavings Rate: {summary['savings_rate']}%"
    return "I couldn't retrieve your weekly statistics. Please make sure you're authenticated."


def get_monthly_stats(assistant, user_id):
    if assistant.stats_api:
        stats = assistant.stats_api.get_monthly_stats()
        if stats and stats.get('success'):
            data = stats['data']
            summary = data['summary']
            return f"📈 Monthly Financial Summary:\nIncome: {summary['total_income']} {data['currency']}\nExpenses: {summary['total_expenses']} {data['currency']}\nNet Flow: {summary['net_flow']} {data['currency']}\nTransactions: {summary['transaction_count']}"
    return "I couldn't retrieve your monthly statistics. Please make sure you're authenticated."


def get_top_categories(assistant, user_id):
    if assistant.stats_api:
        stats = assistant.stats_api.get_top_categories()
        if stats and stats.get('success'):
            data = stats['data']
            if data.get('categories'):
                response = "🎯 Top Spending Categories:\n"
                for i, cat in enumerate(data['categories'][:5], 1):
                    response += f"{i}. {cat['category']}: {cat['amount']} {data['currency']} ({cat['percentage']}%)\n"
                return response
    return "I couldn't retrieve your spending categories. Please make sure you're authenticated."


# Flask app
app = Flask(__name__)

# Configuration
STATS_API_BASE_URL = os.getenv('STATS_API_BASE_URL', 'http://localhost:8000/api')


# Initialize FinancialAssistant
def create_assistant(auth_token=None):
    stats_api = StatisticsAPI(STATS_API_BASE_URL, auth_token or 'default_token')

    # Create function mappings with assistant instance
    financial_functions = {
        'weekly_stats': lambda user_id: get_weekly_stats(assistant, user_id),
        'monthly_stats': lambda user_id: get_monthly_stats(assistant, user_id),
        'top_categories': lambda user_id: get_top_categories(assistant, user_id),
        'debt_snowball': lambda user_id: debt_snowball_calculator(),
        'lebanon_advice': lambda user_id: lebanon_financial_tips()
    }

    assistant = FinancialAssistant('intents.json', function_mappings=financial_functions, stats_api=stats_api)
    return assistant


# Global assistant instance (will be initialized per request with proper token)
assistant = None


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    logging.debug(f"Received request: {data}")

    message = data.get('message')
    user_id = data.get('user_id', 'default_user')
    auth_token = data.get('auth_token')  # Get auth token from request

    if not message:
        logging.error("No message provided")
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Create assistant instance with current user's auth token
        current_assistant = create_assistant(auth_token)

        # Load model if not already loaded
        model_files = ['best_model.pth', 'vectorizer.pkl', 'dimensions.json']
        if all(os.path.exists(f) for f in model_files):
            current_assistant.load_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')
        else:
            logging.error("Model files not found")
            return jsonify({'error': 'Chatbot model not available'}), 500

        response = current_assistant.process_message(message, user_id, auth_token)
        logging.debug(f"Response: {response}")
        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Existing helper functions
def debt_snowball_calculator():
    debts = [
        {"name": "Credit Card", "balance": random.randint(1000, 5000)},
        {"name": "Student Loan", "balance": random.randint(5000, 20000)},
        {"name": "Car Loan", "balance": random.randint(8000, 15000)}
    ]
    debts_sorted = sorted(debts, key=lambda x: x['balance'])
    result = "Debt Snowball Plan:\n"
    for i, debt in enumerate(debts_sorted):
        result += f"{i + 1}. Pay off {debt['name']}: ${debt['balance']:,}\n"
    return result


def lebanon_financial_tips():
    tips = [
        "Convert savings to USD or stable currencies",
        "Prioritize essential goods investments",
        "Use multiple banks for diversification",
        "Consider USDT for dollar exposure",
        "Explore remote work for USD income"
    ]
    return "Lebanon Tips:\n- " + "\n- ".join(random.sample(tips, 3))


if __name__ == '__main__':
    # Train model if needed
    model_files = ['best_model.pth', 'vectorizer.pkl', 'dimensions.json']
    if not all(os.path.exists(f) for f in model_files):
        logging.info("Training new financial model...")
        temp_assistant = FinancialAssistant('intents.json')
        temp_assistant.train_model(epochs=150, batch_size=32)
        temp_assistant.save_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')

    app.run(host='0.0.0.0', port=5001, debug=False)