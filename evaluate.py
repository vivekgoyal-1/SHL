

import pandas as pd
import os
from dotenv import load_dotenv
from typing import Dict, List



def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int = 10) -> float:
    """Calculate Recall@K for a single query"""
    if not ground_truth:
        return 0.0
    
    predictions_k = predictions[:k]
    relevant_retrieved = len(set(predictions_k) & set(ground_truth))
    recall = relevant_retrieved / len(ground_truth)
    
    return recall


def evaluate_system():
    """Evaluate the recommendation system on train data"""
    from recommendation_engine_updated import RecommendationEngineUpdated
    
    load_dotenv()
    
    print("="*70)
    print("EVALUATION: Mean Recall@10 on Train Data")
    print("="*70)
    
    # Initialize engine
    print("\nInitializing recommendation engine...")
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    # Load train data
    print("Loading train data...")
    train_df = pd.read_csv('data/train.csv')
    
    # Group by query
    ground_truth_grouped = train_df.groupby('Query')['Assessment_url'].apply(list).to_dict()
    
    print(f"Evaluating {len(ground_truth_grouped)} queries...")
    print("\n" + "="*70)
    
    recalls = []
    results_details = []
    
    for i, (query, ground_truth) in enumerate(ground_truth_grouped.items(), 1):
        print(f"\nQuery {i}/{len(ground_truth_grouped)}")
        print(f"Query: {query[:65]}...")
        
        # Get recommendations
        recommendations = engine.recommend(query, top_k=10)
        pred_urls = [rec['url'] for rec in recommendations]
        
        # Calculate recall
        recall = calculate_recall_at_k(pred_urls, ground_truth, k=10)
        recalls.append(recall)
        
        # Count matches
        matches = len(set(pred_urls) & set(ground_truth))
        
        results_details.append({
            'query': query,
            'ground_truth_count': len(ground_truth),
            'matches': matches,
            'recall': recall
        })
        
        print(f"  Ground truth: {len(ground_truth)} assessments")
        print(f"  Found in top-10: {matches}")
        print(f"  Recall@10: {recall:.4f}")
    
    # Calculate mean recall
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"Number of queries: {len(recalls)}")
    print(f"Best query recall: {max(recalls):.4f}")
    print(f"Worst query recall: {min(recalls):.4f}")
    print("="*70)
    
    # Save detailed results
    results_df = pd.DataFrame(results_details)
    results_df.to_csv('evaluation_results.csv', index=False)
    print(f"\nDetailed results saved to: evaluation_results.csv")
    
    # Show queries with low recall
    low_recall = results_df[results_df['recall'] < 0.5].sort_values('recall')
    if len(low_recall) > 0:
        print(f"\nQueries with Recall < 0.5: ({len(low_recall)})")
        for _, row in low_recall.iterrows():
            print(f"  - {row['query'][:60]}... (Recall: {row['recall']:.3f})")
    
    return mean_recall, results_details


def generate_test_predictions():
    """Generate predictions for test set"""
    from recommendation_engine_updated import RecommendationEngineUpdated
    
    load_dotenv()
    
    print("="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    # Initialize engine
    print("\nInitializing recommendation engine...")
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    # Load test queries
    print("Loading test queries...")
    test_df = pd.read_csv('data/test.csv')
    queries = test_df['Query'].tolist()
    
    print(f"Generating predictions for {len(queries)} test queries...")
    print("\n" + "="*70)
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}")
        print(f"Query: {query[:65]}...")
        
        # Get recommendations
        recommendations = engine.recommend(query, top_k=10)
        
        print(f"Generated {len(recommendations)} recommendations")
        
        # Add each recommendation as a separate row
        for j, rec in enumerate(recommendations, 1):
            results.append({
                'Query': query,
                'Assessment_url': rec['url']
            })
            print(f"  {j}. {rec['assessment_name'][:55]} (score: {rec['score']:.3f})")
    
    # Save predictions
    print("\n" + "="*70)
    print("Saving predictions...")
    
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv('predictions.csv', index=False)
    
    print(f"✓ Saved predictions.csv")
    print(f"  Total rows: {len(predictions_df)}")
    print(f"  Unique queries: {predictions_df['Query'].nunique()}")
    
    # Validate format
    print("\nValidating format...")
    
    # Check columns
    if list(predictions_df.columns) != ['Query', 'Assessment_url']:
        print("   ERROR: Columns must be ['Query', 'Assessment_url']")
    else:
        print("  Columns correct")
    
    # Check no nulls
    if predictions_df.isnull().any().any():
        print("   ERROR: Contains null values")
    else:
        print("   No null values")
    
    # Check recommendations per query
    recs_per_query = predictions_df.groupby('Query').size()
    print(f"  Recommendations per query: {recs_per_query.min()} to {recs_per_query.max()}")
    
    if recs_per_query.min() < 5 or recs_per_query.max() > 10:
        print("   WARNING: Should be 5-10 recommendations per query")
    
    # Show sample
    print("\nSample (first 10 rows):")
    print(predictions_df.head(10).to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ Predictions generation complete!")
    print("="*70)
    
    return predictions_df


def quick_test():
    """Quick test with sample queries"""
    from recommendation_engine_updated import RecommendationEngineUpdated
    
    load_dotenv()
    
    print("="*70)
    print("QUICK TEST")
    print("="*70)
    
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    test_queries = [
        "Java developer with strong collaboration skills",
        "Python programmer for data analysis",
        "Need personality and cognitive tests for manager"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        recommendations = engine.recommend(query, top_k=10)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. [{rec['test_types']:6s}] {rec['score']:.3f} - {rec['assessment_name'][:50]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'evaluate' or command == 'eval':
            evaluate_system()
        elif command == 'predict' or command == 'test':
            generate_test_predictions()
        elif command == 'quick':
            quick_test()
        else:
            print("Usage:")
            print("  python evaluate.py evaluate  - Run evaluation on train data")
            print("  python evaluate.py predict   - Generate predictions for test data")
            print("  python evaluate.py quick     - Quick test with sample queries")
    else:
        print("="*70)
        print("SHL Assessment Recommendation - Evaluation & Prediction")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            evaluate_system()
        elif choice == '2':
            generate_test_predictions()
        elif choice == '3':
            quick_test()
        else:
            print("Exiting...")


# ==================== run_all.py ====================
"""
Run complete pipeline: evaluate → generate predictions → show results
"""

def run_complete_pipeline():
    """Run the complete evaluation and prediction pipeline"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print("\n" + "="*70)
    print("COMPLETE PIPELINE")
    print("="*70)
    
    # Step 1: Evaluate on train data
    print("\n" + "="*70)
    print("STEP 1: Evaluate on Train Data")
    print("="*70)
    
    mean_recall, details = evaluate_system()
    
    # Step 2: Generate test predictions
    print("\n" + "="*70)
    print("STEP 2: Generate Test Predictions")
    print("="*70)
    
    predictions_df = generate_test_predictions()
    
    # Step 3: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"\n✓ Mean Recall@10 (train): {mean_recall:.4f}")
    print(f"✓ Predictions generated: {len(predictions_df)} rows")
    print(f"✓ Test queries processed: {predictions_df['Query'].nunique()}")
    
    print("\nGenerated files:")
    print("  - evaluation_results.csv (detailed train results)")
    print("  - predictions.csv (test predictions for submission)")
    
    print("\n" + "="*70)
    print("SUBMISSION READY")
    print("="*70)
    


if __name__ == "__main__" and len(__import__('sys').argv) == 1:
    # If run without arguments, run complete pipeline
    run_complete_pipeline()