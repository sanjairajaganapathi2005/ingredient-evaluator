import os
import re
import requests
from dotenv import load_dotenv
import easyocr
from requests.exceptions import RequestException
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    combined_text = " ".join([text for _, text, _ in result])
    return combined_text.strip()


def evaluate_ingredient(ingredients_str):
    
    prompt = (
        "Evaluate the following ingredients individually. "
        "For each one, classify if it's good, moderate, or bad for health, and explain why. "
        "Return the response as a JSON list with keys: 'ingredient', 'evaluation', and 'reason'.\n\n"
        f"Ingredients: {ingredients_str}"
    )
    try:
        result = model.generate_content(prompt)
        return result.text
    except Exception as error:
        print('Error generating response:', error)
        raise

def main():
    image_path = input("Enter the path to the image file: ").strip()

    print("🔍 Extracting text from image...")
    extracted_text = extract_text_from_image(image_path)
    print(f"📝 Extracted text: {extracted_text}")
    ingredients = re.sub(r"[^a-zA-Z0-9, ]+", "", extracted_text)
    if not ingredients:
        print("⚠️ No ingredients found in text.")
        return
    print(f"📋 Ingredients to evaluate: {ingredients}")
    print("🧪 Evaluating ingredients...")

    evaluations = evaluate_ingredient(ingredients)

    print("\n✅ Final Evaluation:")
    print(evaluations)

    
if __name__ == "__main__":
    main()

