import pandas as pd
from recommendation_engine_updated import RecommendationEngineUpdated as RecommendationEngine
import os

def generate_test_predictions(test_file: str, output_file: str):
    """Generate predictions for test set"""
    
    # Initialize engine
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    engine = RecommendationEngine(gemini_api_key=GEMINI_API_KEY)
    
    # Load test queries
    test_df = pd.read_csv(test_file)
    
    results = []
    
    for _, row in test_df.iterrows():
        query = row['Query']
        print(f"Processing: {query[:50]}...")
        
        recommendations = engine.recommend(query, top_k=10)
        
        for rec in recommendations:
            results.append({
                'Query': query,
                'Assessment_url': rec['url']
            })
    
    # Save predictions
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")