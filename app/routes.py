from fastapi import APIRouter
from app.services import get_llm_answer
from app.models import QuestionInput, AnswerOutput

router = APIRouter()

@router.post("/get_answer", response_model=AnswerOutput)
def get_answer(question_input: QuestionInput):
    llm_result = get_llm_answer(question_input.question)
    return AnswerOutput(answer=llm_result['answer'])