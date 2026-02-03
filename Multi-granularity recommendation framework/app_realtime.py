from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import traceback
import sys
from pathlib import Path
import os

from recommendation_framework import MultiGranularityRecommendationFramework


class UserOnboardRequest(BaseModel):
    user_id: str
    age_group: str
    interests: List[str]
    transportation_modes: List[str]
    price_sensitivity: str


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
print("\n" + "=" * 70)
print("LOADING RECOMMENDATION FRAMEWORK...")
print("=" * 70)

try:
    base_dir = Path(__file__).resolve().parent

    def _resolve(p: str) -> str:
        candidates = [
            Path(p),
            base_dir / p,
            base_dir.parent / p,
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        # last resort: search nearby for filename (useful if the file is in a subfolder)
        name = Path(p).name
        for root in [base_dir, base_dir.parent]:
            matches = list(root.rglob(name))
            if matches:
                return str(matches[0])

        # special: if embeddings.pkl isn't present, pick any *embeddings*.pkl nearby
        if name.lower() == "embeddings.pkl":
            for root in [base_dir, base_dir.parent]:
                candidates2 = sorted([x for x in root.rglob("*.pkl") if "embedding" in x.name.lower()])
                if candidates2:
                    return str(candidates2[0])

        # default to base_dir/p for clearer error messages downstream
        return str(base_dir / p)

    framework = MultiGranularityRecommendationFramework(
        embeddings_file=_resolve("embeddings.pkl"),
        interaction_learning_file=_resolve("interaction_learning.pkl"),
        poi_tree_file=_resolve("poi_tree_with_uuids.json"),
        users_file=_resolve("user_preferences.csv"),
        interactions_file=_resolve("user_poi_interactions.csv"),
    )
    print("✅ Framework loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading framework: {e}")
    traceback.print_exc()
    framework = None

print("=" * 70 + "\n")


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
        "total_pois_level_0": len(framework.poi_tree.get("level_0", {})),
    }


@app.post("/users/onboard")
def onboard_user(req: UserOnboardRequest):
    try:
        result = framework.add_user_profile(
            user_id=req.user_id,
            age_group=req.age_group,
            interests=req.interests,
            transportation_modes=req.transportation_modes,
            price_sensitivity=req.price_sensitivity,
            persist=True
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__})


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")

    print("\n" + "=" * 70)
    print("🔍 DETAILED REQUEST DEBUG")
    print("=" * 70)
    print(f"User ID: {request.userId}")
    print(f"Prompt: {request.prompt}")
    print(f"Location: {request.currentLocation}")

    try:
        print("\n[STEP 1] Checking if user exists...")
        if request.userId not in framework.user_id_to_idx:
            available_users = list(framework.user_id_to_idx.keys())[:5]
            error_msg = f"User {request.userId} not found. Onboard first. Example existing: {available_users}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        print(f"✅ User found: {request.userId}")

        print("\n[STEP 2] Generating recommendations...")
        recommendations = framework.recommend_multi_granularity(
            user_id=request.userId,
            levels=[0, 1, 2, 3],
            top_k_per_level=5,
            filter_visited=True,
            prompt=request.prompt,
            current_location=request.currentLocation,
        )
        print("✅ Recommendations generated")

        result = {
            "success": True,
            "userId": request.userId,
            "prompt": request.prompt,
            "recommendations": {},
            "explanations": {},
        }

        for level, recs in recommendations.items():
            result["recommendations"][f"level_{level}"] = []
            result["explanations"][f"level_{level}"] = []

            for (poi_id, score, poi_info) in recs:
                poi_data = framework.poi_tree.get(f"level_{level}", {}).get(poi_id, {})
                poi_spatial = poi_data.get("spatial")

                latitude, longitude = None, None
                if poi_spatial:
                    if isinstance(poi_spatial, str):
                        try:
                            poi_spatial = eval(poi_spatial)
                        except Exception:
                            poi_spatial = None
                    if isinstance(poi_spatial, (list, tuple)) and len(poi_spatial) >= 2:
                        latitude, longitude = poi_spatial[0], poi_spatial[1]

                result["recommendations"][f"level_{level}"].append(
                    {
                        "poi_id": poi_id,
                        "name": poi_info.get("name", "Unknown"),
                        "score": float(score),
                        "type": poi_info.get("type", "Unknown"),
                        "details": {**poi_info, "latitude": latitude, "longitude": longitude},
                    }
                )

                try:
                    explanation = framework.explain_recommendation_enhanced(request.userId, poi_id, level)
                    result["explanations"][f"level_{level}"].append(
                        {
                            "poi_id": poi_id,
                            "poi_name": poi_info.get("name", "Unknown"),
                            "human_explanation": explanation.get("human_explanation", "No explanation"),
                            "top_factors": explanation.get("top_contributing_factors", []),
                            "score": explanation.get("score_breakdown", {}).get("total_score", float(score)),
                        }
                    )
                except Exception:
                    result["explanations"][f"level_{level}"].append(
                        {
                            "poi_id": poi_id,
                            "poi_name": poi_info.get("name", "Unknown"),
                            "human_explanation": "Explanation unavailable",
                            "top_factors": [],
                            "score": float(score),
                        }
                    )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
