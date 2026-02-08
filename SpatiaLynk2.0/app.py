from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import json
import math
import pickle
import traceback
from dataclasses import dataclass
from datetime import datetime
from MPR import MPR_Pipeline 
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Multi-Granularity POI Recommendation API (MPR)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache for recommendations
_REC_CACHE: Dict[str, Dict] = {}
_REC_CACHE_TS: Dict[str, float] = {}
_REC_CACHE_TTL_SECONDS = 300  # 5 minutes


def _make_cache_key(user_id: str, prompt: Optional[str], current_location: Optional[Dict]) -> str:
    p = (prompt or "").strip().lower()
    loc = current_location or {}
    try:
        loc_key = json.dumps(loc, sort_keys=True, ensure_ascii=True)
    except Exception:
        loc_key = str(loc)
    return f"{user_id}|{p}|{loc_key}"

class RecommendationRequest(BaseModel):
    userId: str
    prompt: str
    currentLocation: Optional[Dict[str, float]] = None


class InteractionPayload(BaseModel):
    user_id: str
    poi_id: str
    interaction_type: str
    value: float = 1.0
    timestamp: Optional[str] = None

# Load framework
print("\n" + "="*70)
print("LOADING MPR PIPELINE...")
print("="*70)

def find_repo_root(start=None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    for p in [start] + list(start.parents):
        if (p / ".git").exists():
            return p
    return start

ROOT = find_repo_root()

framework = None
joint_embeddings_path = ROOT / "Sources" / "Embeddings v3" / "joint_optimized_final.pkl"

if joint_embeddings_path.exists():
    try:
        framework = MPR_Pipeline(
            joint_embeddings_file=str(joint_embeddings_path),
            poi_tree_file=str(ROOT / "Sources" / "Files" / "poi_tree_with_uuids.json"),
            users_file=str(ROOT / "Sources" / "Files" / "user_preferences.csv"),
            interactions_file=str(ROOT / "Sources" / "Files" / "user_poi_interactions.csv")
        )
        print("MPR Pipeline loaded successfully!")
        print(f"  Users: {len(framework.users_df)}")
        print(f"  User ID column: {framework.user_id_col}")
    except Exception as e:
        print(f"ERROR loading MPR Pipeline: {e}")
        traceback.print_exc()
        framework = None
else:
    print(f"joint_optimized_final.pkl not found at {joint_embeddings_path}")
    print("Using realtime fallback only.")

print("="*70 + "\n")


SOURCES = ROOT / "Sources"
POI_PKL = SOURCES / "poi_embeddings.pkl"
USER_STORE_PREFIX = SOURCES / "user_vecs" / "user_vecs"
POI_TREE_JSON = SOURCES / "poi_tree_with_uuids.json"


@dataclass(frozen=True)
class InteractionWeights:
    view: float = 0.2
    click: float = 0.6
    visit: float = 1.0
    like: float = 1.2
    search: float = 0.3
    rating: float = 1.0
    bookmark: float = 0.7
    wishlist: float = 0.8
    share: float = 0.9
    review: float = 1.1
    purchase: float = 1.3
    revisit: float = 1.1
    dwell: float = 0.9
    other: float = 0.5


def weight_for(interaction_type: str, value: float, w: InteractionWeights) -> float:
    t = (interaction_type or "").lower().strip()
    if t == "view":
        return w.view
    if t == "click":
        return w.click
    if t == "visit":
        return w.visit
    if t == "like":
        return w.like
    if t == "search":
        return w.search
    if t == "rating":
        v = float(value)
        v = max(1.0, min(5.0, v))
        return (v / 5.0) * w.rating
    if t == "bookmark":
        return w.bookmark
    if t == "wishlist":
        return w.wishlist
    if t == "share":
        return w.share
    if t == "review":
        return w.review
    if t == "purchase":
        return w.purchase
    if t == "revisit":
        return w.revisit
    if t == "dwell":
        v = max(0.0, float(value))
        return min(1.0, v / 300.0) * w.dwell
    return w.other


_TS_FORMATS = (
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
)


def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    s = str(ts).strip()
    if not s:
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


class UserVectorStore:
    def __init__(self, dim: int, prefix: Path):
        self.dim = int(dim)
        self.prefix = Path(prefix)
        self.user_ids = []
        self.id_to_idx: Dict[str, int] = {}
        self.mat = np.zeros((0, self.dim), dtype=np.float32)
        self.dirty = False
        self._load_if_exists()

    @property
    def ids_path(self) -> Path:
        return self.prefix.with_suffix(".json")

    @property
    def mat_path(self) -> Path:
        return self.prefix.with_suffix(".npy")

    def _load_if_exists(self) -> None:
        if self.ids_path.exists() and self.mat_path.exists():
            with self.ids_path.open("r", encoding="utf-8") as f:
                self.user_ids = json.load(f)
            self.id_to_idx = {u: i for i, u in enumerate(self.user_ids)}
            self.mat = np.load(self.mat_path).astype(np.float32, copy=False)

    def get(self, user_id: str) -> np.ndarray:
        idx = self.id_to_idx.get(user_id)
        if idx is None:
            idx = self._add_user(user_id)
        return self.mat[idx]

    def set(self, user_id: str, vec: np.ndarray) -> None:
        idx = self.id_to_idx.get(user_id)
        if idx is None:
            idx = self._add_user(user_id)
        self.mat[idx] = vec.astype(np.float32, copy=False)
        self.dirty = True

    def _add_user(self, user_id: str) -> int:
        idx = len(self.user_ids)
        self.user_ids.append(user_id)
        self.id_to_idx[user_id] = idx
        if self.mat.size == 0:
            self.mat = np.zeros((1, self.dim), dtype=np.float32)
        else:
            self.mat = np.vstack([self.mat, np.zeros((1, self.dim), dtype=np.float32)])
        self.dirty = True
        return idx

    def flush(self) -> None:
        if not self.dirty:
            return
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_ids = self.ids_path.with_suffix(".json.tmp")
        tmp_mat = self.mat_path.with_suffix(".npy.tmp")
        with tmp_ids.open("w", encoding="utf-8") as f:
            json.dump(self.user_ids, f)
        with tmp_mat.open("wb") as f:
            np.save(f, self.mat.astype(np.float32, copy=False))
        tmp_ids.replace(self.ids_path)
        tmp_mat.replace(self.mat_path)
        self.dirty = False


def load_level0_embeddings(pkl_path: Path) -> tuple[Dict[str, np.ndarray], int]:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    lvl0 = obj["poi_embeddings"]["level_0"]
    mat = np.asarray(lvl0["embeddings"], dtype=np.float32)
    ids = lvl0["poi_ids"]
    emb = {str(ids[i]): mat[i] for i in range(mat.shape[0])}
    return emb, int(mat.shape[1])


try:
    _poi_embeddings, _poi_dim = load_level0_embeddings(POI_PKL)
    print("Loaded POI embeddings for interaction API.")
except Exception as e:
    _poi_embeddings, _poi_dim = {}, 0
    print(f"ERROR loading POI embeddings: {e}")

try:
    if POI_TREE_JSON.exists():
        with POI_TREE_JSON.open("r", encoding="utf-8") as f:
            _poi_tree = json.load(f)
    else:
        _poi_tree = {}
except Exception:
    _poi_tree = {}

_weights = InteractionWeights()
_user_store = UserVectorStore(dim=_poi_dim, prefix=USER_STORE_PREFIX) if _poi_dim else None
_last_ts: Dict[str, datetime] = {}
_poi_ids = list(_poi_embeddings.keys())
_poi_mat = np.stack([_poi_embeddings[k] for k in _poi_ids], axis=0) if _poi_ids else None

TIME_DECAY_ENABLED = True
TAU_SECONDS = 7.0 * 24.0 * 3600.0

@app.get("/")
async def root():
    return {
        "message": "Multi-Granularity POI Recommendation API (MPR)",
        "version": "2.0.0",
        "status": "running" if framework else "error",
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "framework_loaded": framework is not None,
        "framework_type": "MPR_Pipeline" if framework else None,
        "total_users": len(framework.users_df) if framework else 0,
        "total_pois_level_0": len(framework.poi_tree.get('level_0', {})) if framework else 0,
        "interaction_ready": bool(_poi_embeddings) and _user_store is not None,
    }


@app.post("/interaction")
async def handle_interaction(payload: InteractionPayload):
    if not _poi_embeddings or _user_store is None:
        raise HTTPException(status_code=503, detail="interaction embeddings not loaded")
    uid = payload.user_id.strip()
    pid = payload.poi_id.strip()
    if not uid or not pid:
        raise HTTPException(status_code=400, detail="missing_user_or_poi")

    vec = _poi_embeddings.get(pid)
    if vec is None:
        raise HTTPException(status_code=404, detail=f"unknown_poi: {pid}")

    alpha = 0.15
    if TIME_DECAY_ENABLED:
        ts_dt = parse_ts(payload.timestamp)
        prev = _last_ts.get(uid)
        if ts_dt is not None and prev is not None:
            dt = (ts_dt - prev).total_seconds()
            if dt > 0 and TAU_SECONDS > 0:
                alpha *= math.exp(-dt / TAU_SECONDS)
        if ts_dt is not None:
            _last_ts[uid] = ts_dt

    w = weight_for(payload.interaction_type, payload.value, _weights)
    u = _user_store.get(uid)
    u2 = (1.0 - alpha) * u + alpha * (w * vec)
    _user_store.set(uid, u2)
    _user_store.flush()

    return {"ok": True, "user_id": uid, "poi_id": pid}

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    # Cache key
    cache_key = _make_cache_key(request.userId, request.prompt, request.currentLocation)
    now_ts = datetime.utcnow().timestamp()
    if cache_key in _REC_CACHE_TS and (now_ts - _REC_CACHE_TS[cache_key]) <= _REC_CACHE_TTL_SECONDS:
        return _REC_CACHE[cache_key]

    if framework is None:
        # Fallback: realtime user vectors + POI embeddings (level_0 only)
        if _user_store is None or _poi_mat is None:
            raise HTTPException(status_code=503, detail="Realtime embeddings not ready")

        uid = request.userId
        if uid not in _user_store.id_to_idx:
            raise HTTPException(status_code=404, detail=f"User {uid} not found in realtime store")

        u = _user_store.get(uid).astype(np.float32)
        un = float(np.linalg.norm(u) + 1e-12)
        v = _poi_mat
        vn = np.linalg.norm(v, axis=1) + 1e-12
        scores = (v @ u) / (vn * un)

        top_k = 10
        top_idx = np.argsort(-scores)[:top_k]
        recs = []
        for i in top_idx:
            pid = _poi_ids[int(i)]
            poi_info = _poi_tree.get("level_0", {}).get(pid, {}) if _poi_tree else {}
            recs.append({
                "poi_id": pid,
                "name": poi_info.get("name", "Unknown"),
                "score": float(scores[int(i)]),
                "details": poi_info,
            })

        resp = {
            "success": True,
            "userId": uid,
            "prompt": request.prompt,
            "recommendations": {"level_0": recs},
            "explanations": {"level_0": []},
            "mode": "realtime_fallback"
        }
        _REC_CACHE[cache_key] = resp
        _REC_CACHE_TS[cache_key] = now_ts
        return resp
    
    print("\n" + "="*70)
    print("DETAILED REQUEST DEBUG")
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
            print(f"ERROR: {error_msg}")
            raise HTTPException(status_code=404, detail=error_msg)
        print(f"User found: {request.userId}")
        
        print("\n[STEP 2] Generating recommendations with MPR_Pipeline...")
        try:
            # You can use top_k=5 for uniform, or a dict for per-level control:
            # top_k={0: 5, 1: 3, 2: 2, 3: 1}
            recommendations = framework.recommend_multi_granularity(
                user_id=request.userId,
                levels=[0, 1, 2, 3],  # All levels as per your example
                top_k={0: 5, 1: 3, 2: 2, 3: 1},  # Dict format for granular control
                prompt=request.prompt,
                current_location=request.currentLocation,
                filter_visited=True
            )
            
            # Log classification results stored by MPR_Pipeline
            if hasattr(framework, 'last_prompt_type'):
                print(f"  Prompt type: {framework.last_prompt_type}")
            if hasattr(framework, 'last_intent') and framework.last_intent:
                print(f"  Primary intent: {framework.last_intent}")
                
            print("Recommendations generated:")
            for level_key, recs in recommendations.items():
                print(f"   {level_key}: {len(recs)} items")
                
        except Exception as e:
            print(f"ERROR generating recommendations: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
        
        # Step 3: Build response
        print("\n[STEP 3] Building response...")
        result = {
            "success": True,
            "userId": request.userId,
            "prompt": request.prompt,
            "recommendations": {},
            "explanations": {}
        }
        
        for level_key, recs in recommendations.items():
            print(f"\n  Processing {level_key}...")
            result["recommendations"][level_key] = []
            result["explanations"][level_key] = []
            
            # Extract level number for coordinate lookup
            level_num = int(level_key.split("_")[1])
            
            for idx, rec in enumerate(recs):
                try:
                    # rec is now a dict with keys: poi_id, name, score, type, details
                    poi_id = rec['poi_id']
                    score = rec['score']
                    poi_info = rec['details']  # Contains score_components, category, etc.
                    
                    # Get coordinates from tree (keep existing logic)
                    poi_data = framework.poi_tree.get(level_key, {}).get(poi_id)
                    if not poi_data:
                        print(f"    Warning: POI {poi_id} not found in tree")
                        continue
                    
                    poi_spatial = poi_data.get('spatial')
                    latitude, longitude = None, None
                    
                    if poi_spatial:
                        if isinstance(poi_spatial, str):
                            try:
                                poi_spatial = eval(poi_spatial)
                            except Exception as e:
                                print(f"    Warning: Error parsing spatial data: {e}")
                        
                        if isinstance(poi_spatial, (list, tuple)) and len(poi_spatial) >= 2:
                            latitude, longitude = poi_spatial[0], poi_spatial[1]
                    
                    # Add recommendation
                    rec_data = {
                        "poi_id": poi_id,
                        "name": rec.get('name', 'Unknown'),  # From top-level of rec
                        "score": float(score),
                        "type": rec.get('type', 'Unknown'),  # From top-level of rec
                        "details": {
                            **poi_info,
                            "latitude": latitude,
                            "longitude": longitude,
                        }
                    }
                    result["recommendations"][level_key].append(rec_data)
                    
                    # Generate explanation if method exists
                    if hasattr(framework, 'generate_explanation_text'):
                        try:
                            explanation = framework.generate_explanation_text(rec, request.userId, level_num)
                            result["explanations"][level_key].append(explanation)
                        except Exception as e:
                            print(f"    Warning: Could not generate explanation: {e}")
                
                except Exception as e:
                    print(f"    ERROR processing recommendation {idx}: {e}")
                    traceback.print_exc()
                    continue
            
            print(f"    Processed {len(result['recommendations'][level_key])} POIs")
        
        print("\n" + "="*70)
        print("REQUEST COMPLETED SUCCESSFULLY")
        print("="*70)
        
        _REC_CACHE[cache_key] = result
        _REC_CACHE_TS[cache_key] = now_ts
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print("\n" + "="*70)
        print("UNEXPECTED ERROR")
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