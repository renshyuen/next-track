# main.py (at project root: next-track/application/main.py)

from fastapi import FastAPI, APIRouter
from application.api.endpoints import recommend

app = FastAPI(
    title="Next-Track: Music Recommendation API",
    summary="Best Music Recommendation API",
    version="1.0.0",
)

app.include_router(recommend.router, prefix='/api')

@app.get('/')
def root():
    return {"message": "test"}

# health check
@app.get('/health')
def health():
    return {"status": "ok"}