"""Debug script to check Sarvam LLM response structure."""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ["SARVAM_API_KEY"]

client = OpenAI(api_key=api_key, base_url="https://api.sarvam.ai/v1")

response = client.chat.completions.create(
    model="sarvam-30b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
        {"role": "user", "content": 'Classify this text: "Today\'s news is very exciting." Respond with JSON: {"emotion": "...", "style": "..."}'},
    ],
    temperature=0.2,
    max_tokens=200,
)

print("=== Full response ===")
print(response)
print()
print("=== choices[0] ===")
print(response.choices[0])
print()
print("=== message ===")
msg = response.choices[0].message
print(f"content type: {type(msg.content)}")
print(f"content: {repr(msg.content)}")
# Check for refusal or other fields
print(f"role: {msg.role}")
if hasattr(msg, 'refusal'):
    print(f"refusal: {msg.refusal}")
