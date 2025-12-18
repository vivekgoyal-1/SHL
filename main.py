import os
from dotenv import load_dotenv

from recommendation_engine_updated import RecommendationEngineUpdated
from scraper_updated import SHLScraperUpdated

if __name__ == "__main__":
    load_dotenv()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    print("="*70)
    print("SHL Assessment Recommendation System - Setup")
    print("="*70)
    
    # Step 1: Load Excel data
    print("\nStep 1: Loading Excel data")
    try:
        from load_from_excel import load_excel_data, prepare_ground_truth
        train_df, test_df = load_excel_data('Gen_AI Dataset.xlsx')
        ground_truth = prepare_ground_truth(train_df)
    except Exception as e:
        print(f"Note: {e}")
    
    # Step 2: Scrape SHL catalog
    print("\nStep 2: Scraping SHL catalog...")
    scraper = SHLScraperUpdated()
    assessments = scraper.scrape_individual_tests()
    print(f" Scraped {len(assessments)} individual test solutions")
    
    
    # Step 3: Initialize recommendation engine
    print("\nStep 3: Initializing recommendation engine...")
    engine = RecommendationEngineUpdated(gemini_api_key=os.getenv('GEMINI_API_KEY'))
    print(" Engine initialized")
    
    # Step 4: Test with sample queries
    print("\nStep 4: Testing with sample queries...")
    
    sample_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams.",
        "Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.",
        "Need cognitive and personality tests for analyst position screening."
    ]
    
    for query in sample_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        recommendations = engine.recommend(query, top_k=10)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['assessment_name'][:60]}")
            print(f"    Score: {rec['score']:.3f} | Types: {rec['test_types']}")
    
    print("\n" + "="*70)
    print("Setup complete!")
    print("="*70)
    