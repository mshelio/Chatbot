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

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)


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


class FinancialAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Financial synonyms dictionary
        self.synonyms = {
            "debt": ["loan", "balance", "owe", "liability"],
            "save": ["set aside", "reserve", "accumulate"],
            "invest": ["put money", "allocate", "grow wealth"],
            "budget": ["spending plan", "financial plan", "allocation"],
            "emergency": ["crisis", "unexpected", "rainy day"]
        }

        self.documents = []
        self.intents = []
        self.intent_responses = {}
        self.function_mappings = function_mappings or {}
        self.context = defaultdict(str)

    def enhanced_tokenize(self, text):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize and lemmatize
        words = nltk.word_tokenize(text.lower())

        # Handle synonyms and lemmatize
        processed_words = []
        for word in words:
            # Replace with base synonym if exists
            for base, syn_list in self.synonyms.items():
                if word in syn_list:
                    word = base
                    break
            # Lemmatize and filter stopwords
            lemma = self.lemmatizer.lemmatize(word)
            if lemma not in self.stop_words and len(lemma) > 2:
                processed_words.append(lemma)

        return processed_words

    def parse_intents(self):
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        all_patterns = []
        intent_labels = []

        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intent_responses[tag] = intent['responses']

            for pattern in intent['patterns']:
                processed_text = " ".join(self.enhanced_tokenize(pattern))
                all_patterns.append(processed_text)
                self.documents.append((processed_text, tag))
                intent_labels.append(tag)

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(all_patterns).toarray()
        self.labels = np.array([self.intents.index(tag) for tag in intent_labels])
        return X

    def prepare_data(self):
        X = self.parse_intents()
        y = self.labels
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

        # Use scheduler without verbose parameter
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_loss = float('inf')
        patience_counter = 0

        print("Training Financial Advice Model...")
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

            # Print learning rate every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")

            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training complete!")

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
        self.model = EnhancedChatbotModel(
            dimensions['input_size'],
            dimensions['output_size']
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message, user_id=None):
        if user_id:
            input_lower = input_message.lower()
            if "lebanon" in input_lower:
                self.context[user_id] = "lebanon"
            elif "debt" in input_lower:
                self.context[user_id] = "debt"
            elif "emergency" in input_lower:
                self.context[user_id] = "emergency"

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
            return "I'm not sure I understand. Could you rephrase your financial question?"

        func_response = ""
        if predicted_intent in self.function_mappings:
            try:
                func_response = self.function_mappings[predicted_intent]()
                if not isinstance(func_response, str):
                    func_response = str(func_response)
            except Exception as e:
                print(f"Function error: {e}")
                func_response = ""

        response = random.choice(self.intent_responses[predicted_intent])

        context = self.context.get(user_id, "")
        if context == "lebanon":
            response += "\n\nFor Lebanon: Consider dollarizing assets and diversifying banks."
        elif context == "debt":
            response += "\n\nRemember: Attack smallest debts first with 'gazelle intensity'!"

        if func_response:
            response += f"\n{func_response}"

        return response


# Financial functions
def get_stocks():
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'V', 'MA', 'PG', 'DIS']
    portfolio = random.sample(stocks, 3)
    return f"Recommended stocks: {', '.join(portfolio)}"


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
    financial_functions = {
        'stocks': get_stocks,
        'debt_snowball': debt_snowball_calculator,
        'lebanon_advice': lebanon_financial_tips
    }

    assistant = FinancialAssistant('intents.json', function_mappings=financial_functions)

    model_files = ['best_model.pth', 'vectorizer.pkl', 'dimensions.json']
    if not all(os.path.exists(f) for f in model_files):
        print("Training new financial model...")
        assistant.train_model(epochs=150, batch_size=32)
        assistant.save_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')
    else:
        print("Loading existing financial model...")
        assistant.load_model('best_model.pth', 'vectorizer.pkl', 'dimensions.json')

    print("Financial Assistant: Hello! How can I help with your finances today? (type '/quit' to exit)")
    user_id = "default_user"

    while True:
        try:
            message = input('\nYou: ')
            if message.lower() == '/quit':
                break

            response = assistant.process_message(message, user_id)
            print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}\nPlease try again.")