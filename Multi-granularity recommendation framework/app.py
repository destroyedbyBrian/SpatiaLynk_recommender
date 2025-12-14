from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import pickle
import json

from recommendation_framework import MultiGranularityRecommendationFramework

app = FastAPI(title="Multi-Granularity POI Recommendation API")

# Load framework once at startup
print("Loading recommendation framework...")
framework = MultiGranularityRecommendationFramework(
    embeddings_file='embeddings.pkl',
    interaction_learning_file='interaction_learning.pkl',
    poi_tree_file='poi_tree_with_uuids.json',
    users_file='user_preferences.csv',
    interactions_file='user_poi_interactions.csv'
)
print("Framework loaded successfully!")

class RecommendationRequest(BaseModel):
    userId: str
    prompt: str
    currentLocation: Optional[Dict[str, float]] = None

@app.get("/")
async def root():
    return {
        "message": "Multi-Granularity POI Recommendation API",
        "endpoints": {
            "POST /recommend": "Get personalized POI recommendations",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "framework_loaded": framework is not None}

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    """
    Generate multi-granularity POI recommendations
    
    Example request:
    {
        "userId": "966592ed-5bfd-4113-9c4d-d93cd3637b40",
        "prompt": "cafes near me",
        "currentLocation": {"latitude": 1.3329, "longitude": 103.7436}
    }
    """
    try:
        # Generate recommendations
        recommendations = framework.recommend_multi_granularity(
            user_id=request.userId,
            levels=[0, 1, 2, 3],
            top_k_per_level=5,
            filter_visited=True
        )
        
        # Generate explanations
        result = {
            "success": True,
            "userId": request.userId,
            "prompt": request.prompt,
            "recommendations": {},
            "explanations": {}
        }
        
        for level, recs in recommendations.items():
            result["recommendations"][f"level_{level}"] = []
            result["explanations"][f"level_{level}"] = []
            
            for poi_id, score, poi_info in recs:
                # Add recommendation
                result["recommendations"][f"level_{level}"].append({
                    "poi_id": poi_id,
                    "name": poi_info['name'],
                    "score": float(score),
                    "type": poi_info['type'],
                    "details": poi_info
                })
                
                # Add explanation
                explanation = framework.explain_recommendation_enhanced(
                    request.userId, poi_id, level
                )
                result["explanations"][f"level_{level}"].append({
                    "poi_id": poi_id,
                    "poi_name": poi_info['name'],
                    "human_explanation": explanation['human_explanation'],
                    "top_factors": explanation['top_contributing_factors'],
                    "score": explanation['score_breakdown']['total_score']
                })
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)