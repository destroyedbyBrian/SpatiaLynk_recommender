"""
Demo v2: Level-2 Explainability

Expected files in the same folder:
- POI.csv
- user_poi_interactions_full.csv

Overall pipeline:
    CSVs
      - derive 1 target user profile from real history
      - mock parsed_intent from NLP+GeoParser
      - build candidate POIs for that user
      - compute reason_flags + score for each POI
      - build short human explanation text
      - print top-k recommendations with explanations

"""

import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path


# =========================
# 0. Config / constants
# =========================

BASE_DIR = Path(__file__).resolve().parent  # folder where explainability.py exist

POI_FILE = BASE_DIR / "POI.csv"
INTER_FILE = BASE_DIR / "user_poi_interactions_full.csv"

# Thresholds / hyperparameters
INTEREST_MATCH_THRESHOLD = 0.45    # how high interest_match_score must be
POPULARITY_THRESHOLD = 4.0         # minimum popularity for "high_popularity"
TOP_CAT_LIMIT = 3                  # how many top categories to store
TOP_K_RESULTS = 5                  # how many final POIs to show


# =========================
# 1. Load data
# =========================

poi_df = pd.read_csv(POI_FILE)
inter_df = pd.read_csv(INTER_FILE)


# =========================
# 2. Pick a user and build a simple profile for testing
# =========================

# Choose a user with the most interactions 
target_user_id = inter_df["user_id"].value_counts().idxmax()

user_hist = inter_df[inter_df["user_id"] == target_user_id].copy()
user_name = user_hist["user_name"].iloc[0]

# Join user's interactions with POI info using name
# (poi_name in interactions vs name in POI dataset)
user_pois = user_hist.merge(
    poi_df,
    left_on="poi_name",
    right_on="name",
    how="inner"
)

# If join is empty, something is wrong with join keys
if user_pois.empty:
    raise ValueError("No POIs matched by name. Check join keys between interactions and POI.csv.")

# Build user "preferences" from history:
# - top categories
# - a comfortable distance radius based on what they usually travel
top_categories = (
    user_pois["category"]
    .value_counts()
    .head(TOP_CAT_LIMIT)
    .index
    .tolist()
)

# Use 75th percentile of distances as "comfortable" radius
max_distance_km = float(user_pois["distance_km"].quantile(0.75))

user: Dict[str, Any] = {
    "user_id": target_user_id,
    "user_name": user_name,
    "preferences": {
        "categories": top_categories,       # e.g. ["restaurant", "cafe", "gym"]
        "max_distance_km": max_distance_km  # typical radius based on past history
    }
}


# =========================
# 3. Mock parsed_intent (output of NLP+GeoParser) (for testing)
# =========================

parsed_intent: Dict[str, Any] = {
    "raw_text": "find a cafe near me that matches my usual interests",
    "categories": ["cafe"],           # NLP detected user asked for cafes
    "keywords": ["quiet", "work"],    # ambience keywords (for future use)
    "dietary": [],                    # none specified in this example
    "spatial_filter": {
        "type": "point_radius",
        "center": None,              # lat/lon
        "radius_km": max_distance_km, # user comfort radius
        "name": "your area"
    },
    "preferred_levels": ["venue"],
    "top_k": TOP_K_RESULTS
}


# =========================
# 4. Build candidate POIs for this user
# =========================

# Aggregate per POI for this user:
# - mean distance
# - mean interest_match_score
# - mean popularity
# - the set of interaction types (visit/search/rating)
candidate_df = (
    user_pois
    .groupby(["poi_id", "poi_name", "category"], as_index=False)
    .agg({
        "distance_km": "mean",
        "interest_match_score": "mean",
        "popularity": "mean",
        "lat": "mean",
        "lon": "mean",
        "district": "first",
        "characteristic": "first",
        "interaction_type": lambda s: ",".join(sorted(set(s)))
    })
)

# Normalisation helpers (avoid division by 0)
max_popularity = max(1.0, float(candidate_df["popularity"].max()))


# =========================
# 5. Reason flags and scoring
# =========================

def build_reason_flags(user: Dict[str, Any],
                       parsed_intent: Dict[str, Any],
                       row: pd.Series) -> Dict[str, Any]:
    """
    Build boolean "reason flags" for a single POI candidate.
    These simple booleans are the core of Level-2 explanations.
    They are later turned into human sentences.
    """
    prefs = user["preferences"]
    sf = parsed_intent["spatial_filter"]

    # Category: either from the current query or from long-term preferences
    match_category_from_intent = row["category"] in parsed_intent["categories"]
    match_category_from_profile = row["category"] in prefs["categories"]

    # Distance: from history (distance_km) vs radius from intent
    max_radius = sf.get("radius_km", prefs["max_distance_km"])
    within_radius_from_intent = row["distance_km"] <= max_radius

    # Behaviour: user has visits/ratings to this POI
    interaction_types = row["interaction_type"].split(",")
    visited_before = "visit" in interaction_types
    rated_before = "rating" in interaction_types

    # Interest match score: higher - more aligned with user preferences
    high_interest_match = row["interest_match_score"] >= INTEREST_MATCH_THRESHOLD

    # Popularity (from POI)
    high_popularity = row["popularity"] >= POPULARITY_THRESHOLD  # using your scale (e.g. 1–5)

    reason_flags = {
        "match_category_from_intent": match_category_from_intent,
        "match_category_from_profile": match_category_from_profile,
        "within_radius_from_intent": within_radius_from_intent,
        "visited_before": visited_before,
        "rated_before": rated_before,
        "high_interest_match": high_interest_match,
        "high_popularity": high_popularity,
    }
    return reason_flags


def score_candidate(user: Dict[str, Any],
                    parsed_intent: Dict[str, Any],
                    row: pd.Series) -> Tuple[float, Dict[str, Any]]:
    """
    Score a POI candidate using:
      - reason flags (booleans)
      - simple weighted sum over factors

    Returns:
        score (float)
        meta dict with:
            - reason_flags (for explanation)
            - rank_meta (numbers for debugging, if needed)
    """
    prefs = user["preferences"]
    rf = build_reason_flags(user, parsed_intent, row)

    # Convert flags to numeric features (0 / 1)
    cat_intent_score = 1.0 if rf["match_category_from_intent"] else 0.0
    cat_profile_score = 1.0 if rf["match_category_from_profile"] else 0.0

    max_d = prefs["max_distance_km"]
    dist_score = max(0.0, 1.0 - (row["distance_km"] / max_d)) if max_d > 0 else 0.0

    # Behaviour score: combine visit/rating into one numeric term
    beh_score = 0.0
    if rf["visited_before"]:
        beh_score += 0.7
    if rf["rated_before"]:
        beh_score += 0.3

    interest_score = float(row["interest_match_score"])
    popularity_score = float(row["popularity"]) / max_popularity

    # Simple weights
    w_cat_intent = 0.25
    w_cat_profile = 0.15
    w_distance = 0.20
    w_behaviour = 0.20
    w_interest = 0.10
    w_popularity = 0.10

    contribution = {
        "category_intent": w_cat_intent * cat_intent_score,
        "category_profile": w_cat_profile * cat_profile_score,
        "distance": w_distance * dist_score,
        "behaviour": w_behaviour * beh_score,
        "interest": w_interest * interest_score,
        "popularity": w_popularity * popularity_score,
    }

    score = sum(contribution.values())

    # For explanation ordering:
    # sort which features contributed most
    sorted_features = sorted(contribution.items(), key=lambda kv: kv[1], reverse=True)
    top_features = [name for name, value in sorted_features if value > 0][:3]

    rank_meta = {
        "score": score,
        "top_features": top_features,
        "distance_km": float(row["distance_km"]),
        "interest_match_score": float(row["interest_match_score"]),
        "popularity": float(row["popularity"]),
    }

    meta = {
        "reason_flags": rf,
        "rank_meta": rank_meta,
    }
    return score, meta


# =========================
# 6. Explanation builder
# =========================

def build_explanation(user: Dict[str, Any],
                      parsed_intent: Dict[str, Any],
                      row: pd.Series,
                      meta: Dict[str, Any]) -> str:
    """
    Turn:
      - reason_flags (booleans)
      - rank_meta (numbers)
      - parsed_intent (what user asked)
    into a short, human explanation for this POI.
    """
    rf = meta["reason_flags"]
    rm = meta["rank_meta"]

    parts: List[str] = []

    # 1) Intro based on intent (what NLP+GeoParser parsed)
    sf = parsed_intent["spatial_filter"]
    loc_name = sf.get("name", "your area")
    intent_cats = parsed_intent.get("categories", [])
    intent_intro = None

    if intent_cats and loc_name:
        intent_intro = f"You asked for {', '.join(intent_cats).lower()} near {loc_name}"
    elif intent_cats:
        intent_intro = f"You asked for {', '.join(intent_cats).lower()}"
    elif loc_name:
        intent_intro = f"You asked for places near {loc_name}"

    # 2) Main reasons in simple language
    if rf.get("match_category_from_intent"):
        parts.append(f"matches your requested category ({row['category']})")
    elif rf.get("match_category_from_profile"):
        parts.append(f"matches a category you often visit ({row['category']})")

    if rf.get("within_radius_from_intent"):
        parts.append(f"is about {rm['distance_km']:.1f} km from you")

    if rf.get("visited_before"):
        parts.append("is a place you have visited before")
    elif rf.get("high_interest_match"):
        parts.append("aligns well with your past interests")

    if rf.get("high_popularity"):
        parts.append("is popular among other users")

    # 3) Build final sentence
    if not parts:
        main = "Recommended based on your past behaviour and location."
    else:
        parts = parts[:3]  # keep explanation short (2–3 reasons)
        main = "Recommended because it " + ", and ".join(parts) + "."

    if intent_intro:
        return intent_intro + ". " + main
    else:
        return main


# =========================
# 7. Run a small demo
# =========================

def main():
    """
    Demo entry point:
    - scores all candidate POIs for this user
    - sorts them by score
    - prints top-k with explanations
    """
    results: List[Dict[str, Any]] = []

    for _, row in candidate_df.iterrows():
        score, meta = score_candidate(user, parsed_intent, row)
        explanation = build_explanation(user, parsed_intent, row, meta)
        results.append({
            "poi_name": row["poi_name"],
            "category": row["category"],
            "district": row["district"],
            "score": meta["rank_meta"]["score"],
            "explanation": explanation,
        })

    # Sort by score (descending) to simulate top-k recommendation
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"User: {user['user_name']} ({user['user_id']})")
    print(f"Preferences (derived): categories={user['preferences']['categories']}, "
          f"max_distance_km={user['preferences']['max_distance_km']:.1f}\n")

    print("=== Top recommendations with explanations ===\n")
    for r in results[:parsed_intent["top_k"]]:
        print(f"- {r['poi_name']}  [{r['category']}] in {r['district']}  (score={r['score']:.3f})")
        print(f"  -> {r['explanation']}\n")


if __name__ == "__main__":
    main()
