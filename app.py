from fastapi import FastAPI
from api import router

app = FastAPI(title="TrackVisionAI API")

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
