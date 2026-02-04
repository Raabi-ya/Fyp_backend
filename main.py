from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
# import openai

app = FastAPI()

# enable CORS to call backnd to React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


label_map = {
    0: "sadeness",
    1: "joy",
    2: "fear",
    3: "anger",
    4: "neutral",
    5: "happy",
    6: "surprise"
}

# Load ML model
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# OpenAI setup
# openai.api_key = "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"  # <-- Replace with your key

class TextRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(message: TextRequest):
    return{"response": f"You said: {message.text}"}
    # try:
      #  response = openai.chat.completions.create(
       #     model="gpt-3.5-turbo",
        #    messages=[{"role": "user", "content": message.text}]
        # )
        # return {"response": response.choices[0].message.content}
    # except Exception as e:
    #    return {"response": f"Error: {str(e)}"}


@app.post("/emotion")
def emotion(text_request: TextRequest):
    # text_lower = text_request.text.lower()
    #if "confused" in text_lower:
     # //  return {"emotion": "fear"}
    text_vec = vectorizer.transform([text_request.text])
    prediction = model.predict(text_vec)[0]
    return {"emotion": label_map[prediction]}

