from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()
from recommendation_engine_updated import RecommendationEngineUpdated as RecommendationEngine

app = FastAPI(title="SHL Assessment Recommendation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
engine = RecommendationEngine(gemini_api_key=GEMINI_API_KEY)

class QueryRequest(BaseModel):
    query: str

class AssessmentRecommendation(BaseModel):
    assessment_name: str
    url: str
    score: Optional[float] = None

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[AssessmentRecommendation]
    count: int

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SHL Assessment Recommendation API",
        "version": "1.0.0"
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: QueryRequest):
    """Get assessment recommendations for a query"""
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        recommendations = engine.recommend(request.query.strip(), top_k=10)
        
        return RecommendationResponse(
            query=request.query,
            recommendations=[
                AssessmentRecommendation(
                    assessment_name=r['assessment_name'],
                    url=r['url'],
                    score=r['score']
                ) for r in recommendations
            ],
            count=len(recommendations)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)