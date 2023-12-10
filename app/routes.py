import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services import get_llm_answer, process_and_index_file, Reset_vector_db_index, crawl_index_website
from app.models import QuestionInput, AnswerOutput, CrawlRequest


router = APIRouter()

@router.post("/api/v1/get_answer", response_model=AnswerOutput)
def get_answer(question_input: QuestionInput):
    llm_result = get_llm_answer(question_input.question)
    return AnswerOutput(answer=llm_result['answer'])

@router.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    file_location = f"{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    #Index File
    process_and_index_file(file_location)

    return {"info": "File uploaded and indexed successfully.", "filename": file.filename}

@router.post("/api/v1/reset_vector_db")
def reset_vector_db():
    
    Reset_vector_db_index()
    return {"message": "VectorDB index has been successfully reset."}
    

@router.post("/crawl_website")
async def crawl_website(request: CrawlRequest):
    url = request.url
    # Implement the logic to crawl the website
    # This might involve calling a function that handles the crawling process
    try:
        # crawl_result = await perform_crawling(url)
        crawl_index_website(url)
        return {"message": "Website indexed successfully."}
    except Exception as e:
        # Catch any exceptions raised during the crawling process
        raise HTTPException(status_code=500, detail=str(e))

