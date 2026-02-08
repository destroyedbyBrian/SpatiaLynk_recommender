from recommendation_framework import MultiGranularityRecommendationFramework

# Load framework
print("Loading framework...")
framework = MultiGranularityRecommendationFramework(
    embeddings_file='embeddings.pkl',
    interaction_learning_file='interaction_learning.pkl',
    poi_tree_file='poi_tree_with_uuids.json',
    users_file='user_preferences.csv',
    interactions_file='user_poi_interactions.csv'
)

print("\n" + "="*70)
print("TEST: Calling compute_multi_granularity_score directly")
print("="*70)

user_id = "6b60d5cf-63cc-4dc4-9bbe-74da03df19db"

# Get first POI
first_poi_id = list(framework.poi_tree['level_0'].keys())[0]

print(f"User: {user_id}")
print(f"POI: {first_poi_id}")
print(f"Level: 0")

# Test 1: Direct call (should work)
try:
    score = framework.compute_multi_granularity_score(user_id, first_poi_id, 0)
    print(f"✅ Direct call SUCCESS: score = {score}")
except Exception as e:
    print(f"❌ Direct call FAILED: {e}")

print("\n" + "="*70)
print("TEST: Calling recommend_at_level")
print("="*70)

# Test 2: Via recommend_at_level (this is where it fails)
try:
    recs = framework.recommend_at_level(
        user_id=user_id,
        level=0,
        top_k=5,
        filter_visited=True,
        use_constraints=False,
        parsed_intent=None
    )
    print(f"✅ recommend_at_level SUCCESS: {len(recs)} recommendations")
except Exception as e:
    print(f"❌ recommend_at_level FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST: Calling recommend_multi_granularity")
print("="*70)

# Test 3: Via recommend_multi_granularity (full flow)
try:
    recs = framework.recommend_multi_granularity(
        user_id=user_id,
        levels=[0],
        top_k_per_level=5,
        filter_visited=True,
        prompt="Cafes Near Me",
        current_location={'latitude': 1.437042896758649, 'longitude': 103.78185926938365}
    )
    print(f"✅ recommend_multi_granularity SUCCESS: {len(recs)} levels")
except Exception as e:
    print(f"❌ recommend_multi_granularity FAILED: {e}")
    import traceback
    traceback.print_exc()