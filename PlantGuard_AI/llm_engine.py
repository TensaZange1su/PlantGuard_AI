import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_llm_explanation(top3_predictions, sensors=None, user_question=None):
    """
    top3_predictions = [
        {"class": "...", "confidence": float},
        ...
    ]
    """

    top1 = top3_predictions[0]
    alternatives = top3_predictions[1:]

    # Format alternative predictions nicely
    alt_text = ""
    for alt in alternatives:
        alt_text += f"- {alt['class']} ({alt['confidence']:.2f}%)\n"

    sensor_text = sensors if sensors else "No sensor data available."

    question_block = f"\nUser Question: {user_question}\n" if user_question else ""

    prompt = f"""
You are an expert agronomist and plant disease specialist.

Plant Vision Model Output:
Top-1:
- Disease: {top1['class']}
- Confidence: {top1['confidence']:.2f}%

Alternative Predictions:
{alt_text}

Sensor Data:
{sensor_text}

{question_block}

Instructions:

1. Evaluate disease likelihood based on confidence.
2. Compare alternatives.
3. Check if sensor data supports environmental stress.
4. Provide:
   - Short diagnosis (2-3 sentences)
   - Risk Level: Low / Medium / High
   - Recommended actions
   - Preventive strategy
5. If user question exists — answer it explicitly.

Keep response professional and concise (150-200 words).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a professional plant pathologist."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content
