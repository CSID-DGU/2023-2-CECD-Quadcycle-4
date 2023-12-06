from fastapi import FastAPI
from pydantic import BaseModel

import GPT2Model
import RoBERTaModel
import Question

app = FastAPI()


@app.get("/api/text")
async def classify_text(text: str):

    # GPT2 모델로 추론
    label = GPT2Model.find_classification(text)

    # Roberta 모델로 추론
    # label = RoBERTaModel.find_classification(text)

    question = Question.find_question(label)

    return {"label": label, "question": question}