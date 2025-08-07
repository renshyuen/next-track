from fastapi import FastAPI, APIRouter
from application.api.endpoints import recommend

app = FastAPI(
    title="Next-Track: Music Recommendation API",
    summary="Best Music Recommendation API",
    version="1.0.0",
)
app.include_router(recommend.router)