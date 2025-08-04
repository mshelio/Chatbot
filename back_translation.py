import json
import logging
from transformers import pipeline
from tqdm import tqdm  # For progress bar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename='paraphrase.log',
    format='%(asctime)s %(levelname)s: %(message)s'
)

# Load paraphrase model
try:
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws",
                           tokenizer="Vamsi/T5_Paraphrase_Paws")
    logging.info("Paraphrase model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load paraphrase model: {str(e)}")
    raise


def para(text, num_return=3, min_length=3):
    """Paraphrase a text and filter low-quality outputs."""
    try:
        prompt = f"paraphrase: {text} </s>"
        outs = paraphraser(prompt, max_length=64, num_return_sequences=num_return)
        paraphrases = [o['generated_text'].strip() for o in outs]
        # Filter paraphrases: remove duplicates, too short, or identical to input
        filtered = [
            p for p in set(paraphrases)
            if len(p.split()) >= min_length and p.lower() != text.lower()
        ]
        logging.info(f"Paraphrased '{text}' into: {filtered}")
        return filtered
    except Exception as e:
        logging.error(f"Error paraphrasing '{text}': {str(e)}")
        return []


def augment_intents(input_file='intents.json', output_file='intents_para.json', max_paraphrases=3):
    """Augment intents.json with paraphrased patterns."""
    try:
        # Load intents
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {input_file} with {len(data['intents'])} intents")

        # Process each intent
        for intent in tqdm(data['intents'], desc="Processing intents"):
            new_patts = []
            for patt in intent['patterns']:
                paraphrases = para(patt, num_return=max_paraphrases)
                new_patts.extend(paraphrases)
            # Combine original and paraphrased patterns, remove duplicates
            intent['patterns'] = list(dict.fromkeys(intent['patterns'] + new_patts))

        # Save augmented intents
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved paraphrased intents to {output_file}")
        print(f"Paraphrased intents saved to {output_file}")

    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        print(f"Error: {input_file} not found")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in {input_file}")
        print(f"Error: Invalid JSON in {input_file}")
    except Exception as e:
        logging.error(f"Error augmenting intents: {str(e)}")
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    augment_intents()