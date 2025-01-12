from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()


def generate():
    client = genai.Client(
        vertexai=True,
        project=os.getenv("VERTEX_PROJECT_ID"),
        location=os.getenv("VERTEX_PROJECT_REGION"),
    )

    text1 = types.Part.from_text(
        """I'm going to Ireland for two weeks in September but I also want to go to Versailles. How can I make this happen and what are my options?"""
    )
    textsi_1 = """You are a travel itinerary generator for European tourists.
Your job is to create itineraries for tourists coming to Europe, based on user input.
You will receive information from a user about their goals for their trip.
You will generate an in-depth travel itinerary for users.
Do not use the internet.
Do not hallucinate.
You must address all of what the user provides.
You must have a catchy title.
You must bullet point every sentence.
Your output must be in a logical format and order.
Address the user by name if it is given.
Come up with at least 2 ideas the user may not have mentioned.
Output 1-2 lines of a history of each place suggested.
Provide 1-2 lines of current cultural background on the place provided.
Provide 1-2 lines of weather related advice for each place suggested.
Provide one line of rationale for each place suggested.
Include one joke in the itinerary.
End the itinerary with a clever, place-related goodbye."""

    model = "gemini-2.0-flash-exp"
    contents = [types.Content(role="user", parts=[text1])]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        max_output_tokens=1024,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        system_instruction=[types.Part.from_text(textsi_1)],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk, end="")


generate()
