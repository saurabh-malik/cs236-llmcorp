from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import get_llm_answer, process_and_index_file
from app.models import QuestionInput, AnswerOutput
import os

router = APIRouter()

@router.post("/get_answer", response_model=AnswerOutput)
def get_answer(question_input: QuestionInput):
    llm_result = get_llm_answer(question_input.question)
    return AnswerOutput(answer=llm_result['answer'])

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    file_location = f"{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    #Index File
    process_and_index_file(file_location)

    return {"info": "File uploaded and indexed successfully.", "filename": file.filename}