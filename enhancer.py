import json

with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

print("Intent coverage:")
for intent in data['intents']:
    tag = intent['tag']
    count = len(intent['patterns'])
    print(f"  • {tag:<20} → {count} examples")
