from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from recommendation_framework_realtime import MultiGranularityRecommendationFramework


class UserOnboardRequest(BaseModel):
    user_id: str
    age_group: str
    interests: List[str]
    transportation_modes: List[str]
    price_sensitivity: str


class RecommendationRequest(BaseModel):
    userId: str
    prompt: str
    currentLocation: Optional[Dict[str, float]] = None


app = FastAPI(
    title="Multi-Granularity POI Recommendation API (Realtime)",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

framework: Optional[MultiGranularityRecommendationFramework] = None


def _resolve(base_dir: Path, p: str) -> str:
    candidates = [
        Path(p),
        base_dir / p,
        base_dir.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    name = Path(p).name
    for root in [base_dir, base_dir.parent]:
        matches = list(root.rglob(name))
        if matches:
            return str(matches[0])

    if name.lower() == "embeddings.pkl":
        for root in [base_dir, base_dir.parent]:
            candidates2 = sorted([x for x in root.rglob("*.pkl") if "embedding" in x.name.lower()])
            if candidates2:
                return str(candidates2[0])

    return str(base_dir / p)


print("\n" + "=" * 70)
print("LOADING RECOMMENDATION FRAMEWORK (REALTIME, LOCATION-AWARE)...")
print("=" * 70)

try:
    base_dir = Path(__file__).resolve().parent
    framework = MultiGranularityRecommendationFramework(
        embeddings_file=_resolve(base_dir, "embeddings.pkl"),
        interaction_learning_file=_resolve(base_dir, "interaction_learning.pkl"),
        poi_tree_file=_resolve(base_dir, "poi_tree_with_uuids.json"),
        users_file=_resolve(base_dir, "user_preferences.csv"),
        interactions_file=_resolve(base_dir, "user_poi_interactions.csv"),
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
        "message": "Multi-Granularity POI Recommendation API (Realtime)",
        "status": "running" if framework else "error",
    }


@app.get("/health")
async def health_check():
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")
    return framework.health()


@app.post("/users/onboard")
def onboard_user(req: UserOnboardRequest):
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")

    profile = {
        "user_id": req.user_id,
        "age_group": req.age_group,
        "interests": req.interests,
        "transportation_modes": req.transportation_modes,
        "price_sensitivity": req.price_sensitivity,
    }
    result = framework.add_user_profile(profile)
    return {"status": "ok", "result": result}


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    if framework is None:
        raise HTTPException(status_code=503, detail="Framework not loaded")

    # Store last location if provided
    if request.currentLocation:
        framework.user_profiles.setdefault(request.userId, {})
        framework.user_profiles[request.userId]["last_location"] = request.currentLocation

    try:
        recs = framework.recommend_multi_granularity(
            user_id=request.userId,
            top_k=10,
            prompt=request.prompt,
            use_intent_filter=True,
            current_location=request.currentLocation,  # NEW
        )
        return {
            "success": True,
            "userId": request.userId,
            "prompt": request.prompt,
            "recommendations": recs,
            "explanations": {k: [] for k in recs.keys()},
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
