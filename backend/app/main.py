from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.registry import ModelRegistry
from app.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    app.state.registry.load_all()
    yield


app = FastAPI(title="MODI Recognition API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://akshar-demo-gauri.s3-website.ap-south-1.amazonaws.com"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
