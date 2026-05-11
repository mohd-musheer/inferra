from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware 

import requests

app = FastAPI()

app.add_middleware(
    
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")


class PromptRequest(BaseModel):
    prompt: str


LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/generate")
async def generate(data: PromptRequest):

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Format responses using markdown. "
                    "Use headings, bullet points, and code blocks when needed."
                )
            },
            {
                "role": "user",
                "content": data.prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(
        LLAMA_SERVER_URL,
        json=payload
    )

    result = response.json()

    text = result["choices"][0]["message"]["content"]

    return JSONResponse({
        "response": text
    })