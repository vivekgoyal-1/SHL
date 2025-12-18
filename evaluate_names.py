#!/usr/bin/env python3
"""
Name-based evaluation - matches assessments by name instead of URL
This solves the URL mismatch problem
"""

import pandas as pd
import json
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher

def normalize_name(name):
    """Normalize assessment name for comparison"""
    # Convert to lowercase, remove special chars, normalize spaces
    name = name.lower().strip()
    # Remove common variations
    name = name.replace('(new)', '').replace('- short form', '').replace('-', ' ')
    name = ' '.join(name.split())  # Normalize whitespace
    return name

def name_similarity(name1, name2):
    """Calculate similarity between two names"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def url_to_name(url):
    """Extract assessment name from URL"""
    # Get last part of URL
    slug = url.rstrip('/').split('/')[-1]
    # Convert slug to name
    name = slug.replace('-', ' ').title()
    return name

def evaluate_with_name_matching():
    """Evaluate using name-based matching instead of URL matching"""
    from recommendation_engine_updated import RecommendationEngineUpdated
    
    load_dotenv()
    
    print("="*70)
    print("NAME-BASED EVALUATION (URL Independent)")
    print("="*70)
    
    # Initialize engine
    print("\nInitializing recommendation engine...")
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    # Load train data and extract names
    print("Loading train data...")
    train_df = pd.read_csv('data/train.csv')
    
    # Extract assessment names from URLs
    train_df['gt_name'] = train_df['Assessment_url'].apply(url_to_name)
    
    # Group by query
    ground_truth_by_query = {}
    for query, group in train_df.groupby('Query'):
        gt_names = group['gt_name'].tolist()
        ground_truth_by_query[query] = gt_names
    
    print(f"Evaluating {len(ground_truth_by_query)} queries...")
    print("\n" + "="*70)
    
    recalls = []
    results_details = []
    
    for i, (query, gt_names) in enumerate(ground_truth_by_query.items(), 1):
        print(f"\nQuery {i}/{len(ground_truth_by_query)}")
        print(f"Query: {query[:65]}...")
        
        # Get recommendations
        recommendations = engine.recommend(query, top_k=10)
        pred_names = [rec['assessment_name'] for rec in recommendations]
        
        # Calculate matches using fuzzy name matching
        matches = 0
        matched_gt = set()
        
        for pred_name in pred_names:
            for gt_name in gt_names:
                if gt_name not in matched_gt:
                    similarity = name_similarity(pred_name, gt_name)
                    if similarity > 0.7:  # 70% similarity threshold
                        matches += 1
                        matched_gt.add(gt_name)
                        print(f"  ✓ Match: {pred_name} ≈ {gt_name} ({similarity:.2f})")
                        break
        
        recall = matches / len(gt_names) if gt_names else 0
        recalls.append(recall)
        
        results_details.append({
            'query': query,
            'ground_truth_count': len(gt_names),
            'matches': matches,
            'recall': recall
        })
        
        print(f"  Ground truth: {len(gt_names)} assessments")
        print(f"  Found in top-10: {matches}")
        print(f"  Recall@10: {recall:.4f}")
    
    # Calculate mean recall
    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS (Name-Based Matching)")
    print("="*70)
    print(f"Mean Recall@10: {mean_recall:.4f}")
    print(f"Number of queries: {len(recalls)}")
    print(f"Best query recall: {max(recalls):.4f}")
    print(f"Worst query recall: {min(recalls):.4f}")
    print("="*70)
    
    # Save results
    results_df = pd.DataFrame(results_details)
    results_df.to_csv('evaluation_results_name_based.csv', index=False)
    print(f"\nDetailed results saved to: evaluation_results_name_based.csv")
    
    return mean_recall, results_details


def show_ground_truth_vs_scraped():
    """Show what's in ground truth vs what we scraped"""
    print("="*70)
    print("GROUND TRUTH VS SCRAPED ASSESSMENTS")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    train_df['gt_name'] = train_df['Assessment_url'].apply(url_to_name)
    
    with open('data/assessments.json', 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    gt_names = set(normalize_name(n) for n in train_df['gt_name'].unique())
    scraped_names = set(normalize_name(a['name']) for a in assessments)
    
    print(f"\nUnique assessment names in ground truth: {len(gt_names)}")
    print(f"Total assessments scraped: {len(scraped_names)}")
    print(f"Exact name matches: {len(gt_names & scraped_names)}")
    
    # Show missing assessments
    missing = gt_names - scraped_names
    if missing:
        print(f"\nAssessments in ground truth but not scraped: {len(missing)}")
        for name in list(missing)[:10]:
            print(f"  - {name}")
            
            # Try to find close matches
            close_matches = []
            for scraped in scraped_names:
                sim = SequenceMatcher(None, name, scraped).ratio()
                if sim > 0.6:
                    close_matches.append((scraped, sim))
            
            if close_matches:
                close_matches.sort(key=lambda x: x[1], reverse=True)
                print(f"    Close match: {close_matches[0][0]} ({close_matches[0][1]:.2f})")
    
    # Show some scraped assessments
    print(f"\nSample scraped assessments:")
    for a in assessments[:10]:
        print(f"  - {a['name']} (Types: {a.get('test_types', 'N/A')})")


def quick_test_one_query():
    """Quick test on one query to see name matching"""
    from recommendation_engine_updated import RecommendationEngineUpdated
    
    load_dotenv()
    
    print("="*70)
    print("QUICK TEST - One Query with Name Matching")
    print("="*70)
    
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    
    # Load one query from train
    train_df = pd.read_csv('data/train.csv')
    query = train_df.iloc[0]['Query']
    
    # Get ground truth names for this query
    gt_urls = train_df[train_df['Query'] == query]['Assessment_url'].tolist()
    gt_names = [url_to_name(url) for url in gt_urls]
    
    print(f"\nQuery: {query[:70]}...")
    print(f"\nGround truth ({len(gt_names)} assessments):")
    for i, name in enumerate(gt_names, 1):
        print(f"  {i}. {name}")
    
    # Get recommendations
    print(f"\nRecommendations:")
    recommendations = engine.recommend(query, top_k=10)
    
    matches = 0
    for i, rec in enumerate(recommendations, 1):
        # Check if matches any ground truth
        is_match = False
        for gt_name in gt_names:
            similarity = name_similarity(rec['assessment_name'], gt_name)
            if similarity > 0.7:
                is_match = True
                matches += 1
                break
        
        marker = "✓" if is_match else " "
        print(f"  {marker} {i}. {rec['assessment_name'][:60]}")
        print(f"      Score: {rec['score']:.3f} | Types: {rec['test_types']}")
    
    recall = matches / len(gt_names) if gt_names else 0
    print(f"\nRecall@10: {recall:.3f} ({matches}/{len(gt_names)} matched)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test_one_query()
    elif len(sys.argv) > 1 and sys.argv[1] == 'compare':
        show_ground_truth_vs_scraped()
    else:
        print("\n" + "="*70)
        print(" 1: Show what's in ground truth vs scraped")
        print("="*70)
        show_ground_truth_vs_scraped()
        
        print("\n" + "="*70)
        print(" 2: Quick test with one query")
        print("="*70)
        quick_test_one_query()
        
        print("\n" + "="*70)
        print(" 3: Full evaluation with name-based matching")
        print("="*70)
        
        choice = input("\nRun full evaluation? [y/N]: ")
        if choice.lower() == 'y':
            mean_recall, details = evaluate_with_name_matching()
            
           
            print(f"\n Mean Recall@10: {mean_recall:.4f}")
            