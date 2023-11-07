from pydantic import BaseModel

class QuestionInput(BaseModel):
    question: str

class AnswerOutput(BaseModel):
    answer: str
