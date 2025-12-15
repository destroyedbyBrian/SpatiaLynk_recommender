from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import traceback
import sys

from recommendation_framework import MultiGranularityRecommendationFramework

app = FastAPI(
    title="Multi-Granularity POI Recommendation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    userId: str
    prompt: str
    currentLocation: Optional[Dict[str, float]] = None

# Load framework
print("\n" + "="*70)
print("LOADING RECOMMENDATION FRAMEWORK...")
print("="*70)

try:
    framework = MultiGranularityRecommendationFramework(
        embeddings_file='embeddings.pkl',
        interaction_learning_file='interaction_learning.pkl',
        poi_tree_file='poi_tree_with_uuids.json',
        users_file='user_preferences.csv',
        interactions_file='user_poi_interactions.csv'
    )
    print("✅ Framework loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading framework: {e}")
    traceback.print_exc()
    framework = None

print("="*70 + "\n")

@app.get("/")
async def root():
    return {
        "message": "Multi-Granularity POI Recommendation API",
        "version": "1.0.0",
        "status": "running" if framework else "error",
    }

@app.get("/health")
async def health_check():
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")
    
    return {
        "status": "healthy",
        "framework_loaded": True,
        "total_users": len(framework.users_df),
        "total_pois_level_0": len(framework.poi_tree.get('level_0', {})),
    }

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")
    
    # ========================================================================
    # DETAILED ERROR LOGGING
    # ========================================================================
    
    print("\n" + "="*70)
    print("🔍 DETAILED REQUEST DEBUG")
    print("="*70)
    print(f"User ID: {request.userId}")
    print(f"Prompt: {request.prompt}")
    print(f"Location: {request.currentLocation}")
    
    try:
        # Step 1: Check user exists
        print("\n[STEP 1] Checking if user exists...")
        if request.userId not in framework.user_id_to_idx:
            available_users = list(framework.user_id_to_idx.keys())[:5]
            error_msg = f"User {request.userId} not found. Available users: {available_users}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        print(f"✅ User found: {request.userId}")
        
        # Step 2: Parse prompt
        print("\n[STEP 2] Parsing prompt...")
        parsed_intent = None
        if request.prompt:
            try:
                parsed_intent = framework.parse_user_prompt(request.prompt, request.currentLocation)
                print(f"✅ Parsed intent:")
                print(f"   Categories: {parsed_intent.get('categories')}")
                print(f"   Location mentioned: {parsed_intent.get('location_mentioned')}")
            except Exception as e:
                print(f"⚠️ Error parsing prompt: {e}")
                traceback.print_exc()
        
        # Step 3: Generate recommendations
        print("\n[STEP 3] Generating recommendations...")
        try:
            recommendations = framework.recommend_multi_granularity(
                user_id=request.userId,
                levels=[0, 1, 2, 3],
                top_k_per_level=5,
                filter_visited=True,
                prompt=request.prompt,
                current_location=request.currentLocation
            )
            print(f"✅ Recommendations generated:")
            for level, recs in recommendations.items():
                print(f"   Level {level}: {len(recs)} POIs")
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
        
        # Step 4: Build response
        print("\n[STEP 4] Building response...")
        result = {
            "success": True,
            "userId": request.userId,
            "prompt": request.prompt,
            "recommendations": {},
            "explanations": {}
        }
        
        for level, recs in recommendations.items():
            print(f"\n  Processing level {level}...")
            result["recommendations"][f"level_{level}"] = []
            result["explanations"][f"level_{level}"] = []
            
            for idx, (poi_id, score, poi_info) in enumerate(recs):
                try:
                    # Get coordinates
                    poi_data = framework.poi_tree[f'level_{level}'].get(poi_id)
                    if not poi_data:
                        print(f"    ⚠️ POI {poi_id} not found in tree")
                        continue
                    
                    poi_spatial = poi_data.get('spatial')
                    latitude, longitude = None, None
                    
                    if poi_spatial:
                        if isinstance(poi_spatial, str):
                            try:
                                poi_spatial = eval(poi_spatial)
                            except Exception as e:
                                print(f"    ⚠️ Error parsing spatial data: {e}")
                        
                        if isinstance(poi_spatial, (list, tuple)) and len(poi_spatial) >= 2:
                            latitude, longitude = poi_spatial[0], poi_spatial[1]
                    
                    # Add recommendation
                    rec_data = {
                        "poi_id": poi_id,
                        "name": poi_info.get('name', 'Unknown'),
                        "score": float(score),
                        "type": poi_info.get('type', 'Unknown'),
                        "details": {
                            **poi_info,
                            "latitude": latitude,
                            "longitude": longitude,
                        }
                    }
                    result["recommendations"][f"level_{level}"].append(rec_data)
                    
                    # Add explanation
                    try:
                        explanation = framework.explain_recommendation_enhanced(
                            request.userId, poi_id, level
                        )
                        result["explanations"][f"level_{level}"].append({
                            "poi_id": poi_id,
                            "poi_name": poi_info.get('name', 'Unknown'),
                            "human_explanation": explanation.get('human_explanation', 'No explanation'),
                            "top_factors": explanation.get('top_contributing_factors', []),
                            "score": explanation.get('score_breakdown', {}).get('total_score', float(score))
                        })
                    except Exception as e:
                        print(f"    ⚠️ Error generating explanation for {poi_info.get('name')}: {e}")
                        result["explanations"][f"level_{level}"].append({
                            "poi_id": poi_id,
                            "poi_name": poi_info.get('name', 'Unknown'),
                            "human_explanation": "Explanation unavailable",
                            "top_factors": [],
                            "score": float(score)
                        })
                
                except Exception as e:
                    print(f"    ❌ Error processing POI {poi_id}: {e}")
                    traceback.print_exc()
                    continue
            
            print(f"    ✅ Processed {len(result['recommendations'][f'level_{level}'])} POIs")
        
        print("\n" + "="*70)
        print("✅ REQUEST COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print("\n" + "="*70)
        print("❌ UNEXPECTED ERROR")
        print("="*70)
        print(f"Error: {str(e)}")
        print(f"Type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*70)
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)