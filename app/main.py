from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from app.routes import router
from fastapi.responses import HTMLResponse 
from fastapi.staticfiles import StaticFiles  # Import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

# Mount the "static" directory to serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/data-upload", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("data-index.html", {"request": request})

app.include_router(router)