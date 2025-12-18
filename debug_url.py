#!/usr/bin/env python3
"""
Debug and fix URL mismatch between ground truth and scraped data
"""

import pandas as pd
import json
from difflib import get_close_matches
from urllib.parse import urlparse

def analyze_url_mismatch():
    """
    Analyze why ground truth URLs don't match scraped URLs
    """
    print("="*70)
    print("URL MISMATCH ANALYSIS")
    print("="*70)
    
    # Load train data (ground truth)
    train_df = pd.read_csv('data/train.csv')
    gt_urls = set(train_df['Assessment_url'].unique())
    
    # Load scraped assessments
    with open('data/assessments.json', 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    scraped_urls = set(a['url'] for a in assessments)
    
    print(f"\nGround truth URLs: {len(gt_urls)}")
    print(f"Scraped URLs: {len(scraped_urls)}")
    print(f"Exact matches: {len(gt_urls & scraped_urls)}")
    
    # Show sample of each
    print("\nSample ground truth URLs:")
    for url in list(gt_urls)[:3]:
        print(f"  {url}")
    
    print("\nSample scraped URLs:")
    for url in list(scraped_urls)[:3]:
        print(f"  {url}")
    
    # Check for URL pattern differences
    print("\n" + "="*70)
    print("URL PATTERN ANALYSIS")
    print("="*70)
    
    gt_sample = list(gt_urls)[0]
    scraped_sample = list(scraped_urls)[0]
    
    print(f"\nGround truth pattern: {gt_sample}")
    print(f"Scraped pattern:      {scraped_sample}")
    
    # Parse URLs to find differences
    gt_parsed = urlparse(gt_sample)
    scraped_parsed = urlparse(scraped_sample)
    
    print(f"\nDomain: GT={gt_parsed.netloc}, Scraped={scraped_parsed.netloc}")
    print(f"Path: GT={gt_parsed.path}, Scraped={scraped_parsed.path}")
    
    # Find close matches
    print("\n" + "="*70)
    print("FINDING CLOSE MATCHES")
    print("="*70)
    
    print("\nTrying to match ground truth to scraped URLs...")
    matches_found = 0
    
    for gt_url in list(gt_urls)[:5]:
        # Get assessment name from URL
        gt_name = gt_url.split('/')[-2] if gt_url.endswith('/') else gt_url.split('/')[-1]
        gt_name = gt_name.replace('-', ' ').lower()
        
        # Find in scraped assessments
        for assessment in assessments:
            scraped_name = assessment['name'].lower()
            scraped_url = assessment['url']
            
            # Check if names match
            if gt_name in scraped_name or scraped_name in gt_name:
                print(f"\n✓ Match found!")
                print(f"  GT URL:      {gt_url}")
                print(f"  Scraped URL: {scraped_url}")
                print(f"  Name:        {assessment['name']}")
                matches_found += 1
                break
    
    print(f"\nMatches found by name: {matches_found}/5")
    
    return gt_urls, scraped_urls, assessments


def create_url_mapping():
    """
    Create a mapping between ground truth URLs and scraped URLs
    """
    print("\n" + "="*70)
    print("CREATING URL MAPPING")
    print("="*70)
    
    train_df = pd.read_csv('data/train.csv')
    
    with open('data/assessments.json', 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    # Create lookup by name
    name_to_url = {}
    for a in assessments:
        name_lower = a['name'].lower()
        name_to_url[name_lower] = a['url']
    
    # Try to map URLs
    mapping = {}
    unmatched = []
    
    for gt_url in train_df['Assessment_url'].unique():
        # Extract name from URL
        url_parts = gt_url.rstrip('/').split('/')
        url_name = url_parts[-1].replace('-', ' ').lower()
        
        # Try exact match
        if url_name in name_to_url:
            mapping[gt_url] = name_to_url[url_name]
        else:
            # Try fuzzy match
            matches = get_close_matches(url_name, name_to_url.keys(), n=1, cutoff=0.6)
            if matches:
                mapping[gt_url] = name_to_url[matches[0]]
            else:
                unmatched.append(gt_url)
    
    print(f"\nMapped: {len(mapping)} URLs")
    print(f"Unmatched: {len(unmatched)} URLs")
    
    if unmatched:
        print("\nUnmatched URLs (sample):")
        for url in unmatched[:5]:
            print(f"  {url}")
    
    return mapping


def fix_train_data_urls():
    """
    Fix train.csv by mapping ground truth URLs to actual scraped URLs
    """
    print("\n" + "="*70)
    print("FIXING TRAIN DATA URLS")
    print("="*70)
    
    train_df = pd.read_csv('data/train.csv')
    
    with open('data/assessments.json', 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    # Create comprehensive lookup
    url_lookup = {}
    name_lookup = {}
    
    for a in assessments:
        # Store by URL
        url_lookup[a['url'].lower()] = a
        
        # Store by normalized name
        name = a['name'].lower().strip()
        name_normalized = ''.join(c for c in name if c.isalnum() or c.isspace())
        name_lookup[name_normalized] = a
        
        # Also store with URL slug
        url_slug = a['url'].rstrip('/').split('/')[-1]
        name_lookup[url_slug.replace('-', '')] = a
    
    # Fix URLs in train data
    fixed_rows = []
    fixed_count = 0
    unfixed_count = 0
    
    for _, row in train_df.iterrows():
        query = row['Query']
        gt_url = row['Assessment_url']
        
        # Try exact match first
        if gt_url.lower() in url_lookup:
            fixed_rows.append({
                'Query': query,
                'Assessment_url': gt_url  # Already correct
            })
            fixed_count += 1
            continue
        
        # Try name matching
        # Extract name from GT URL
        url_parts = gt_url.rstrip('/').split('/')
        url_name = url_parts[-1] if url_parts else ''
        url_name_normalized = url_name.replace('-', '').lower()
        
        matched = False
        
        # Try to find in lookup
        if url_name_normalized in name_lookup:
            matched_assessment = name_lookup[url_name_normalized]
            fixed_rows.append({
                'Query': query,
                'Assessment_url': matched_assessment['url']
            })
            fixed_count += 1
            matched = True
        else:
            # Try fuzzy matching on assessment names
            best_match = None
            best_score = 0
            
            for norm_name, assessment in name_lookup.items():
                # Calculate simple similarity
                common = sum(1 for c in url_name_normalized if c in norm_name)
                score = common / max(len(url_name_normalized), len(norm_name))
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = assessment
            
            if best_match:
                fixed_rows.append({
                    'Query': query,
                    'Assessment_url': best_match['url']
                })
                fixed_count += 1
                matched = True
        
        if not matched:
            # Keep original but mark as unfixed
            fixed_rows.append({
                'Query': query,
                'Assessment_url': gt_url
            })
            unfixed_count += 1
    
    # Save fixed train data
    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv('data/train_fixed.csv', index=False)
    
    print(f"\n✓ Fixed train data saved to train_fixed.csv")
    print(f"  Fixed: {fixed_count} URLs")
    print(f"  Unfixed: {unfixed_count} URLs")
    print(f"  Total: {len(fixed_rows)} rows")
    
    # Show sample fixes
    if fixed_count > 0:
        print("\nSample fixes:")
        for i in range(min(3, len(train_df))):
            if train_df.iloc[i]['Assessment_url'] != fixed_df.iloc[i]['Assessment_url']:
                print(f"\nQuery: {train_df.iloc[i]['Query'][:50]}...")
                print(f"  Before: {train_df.iloc[i]['Assessment_url']}")
                print(f"  After:  {fixed_df.iloc[i]['Assessment_url']}")
    
    return fixed_df


def simple_solution_use_scraped_only():
    """
    SIMPLE SOLUTION: Just use the scraped assessments and ignore ground truth URLs
    This is because your ground truth URLs might not exactly match the scraped URLs
    """
    print("\n" + "="*70)
    print("SIMPLE SOLUTION: Building from scraped data only")
    print("="*70)
    
    # The key insight: We should evaluate based on ASSESSMENT NAMES not URLs
    # Because URLs might have slight differences
    
    print("""
This is the real issue: Your train.csv has URLs that don't exactly match
the scraped catalog URLs. This could be because:
1. URLs changed on SHL website
2. URL format differences (trailing slash, etc.)
3. Different catalog sections

SOLUTION: Match by assessment NAME instead of URL
    """)
    
    # Create name-based evaluation
    train_df = pd.read_csv('data/train.csv')
    
    with open('data/assessments.json', 'r', encoding='utf-8') as f:
        assessments = json.load(f)
    
    # Extract names from URLs in train data
    print("\nExtracting assessment names from ground truth URLs...")
    
    train_df['assessment_name_from_url'] = train_df['Assessment_url'].apply(
        lambda url: url.rstrip('/').split('/')[-1].replace('-', ' ').title()
    )
    
    print("\nSample extracted names:")
    print(train_df[['Query', 'assessment_name_from_url']].head())
    
    # Show what we have
    print(f"\nUnique assessment names in ground truth: {train_df['assessment_name_from_url'].nunique()}")
    print(f"Total assessments scraped: {len(assessments)}")
    
    # Find matches
    scraped_names = {a['name'].lower(): a for a in assessments}
    gt_names = set(train_df['assessment_name_from_url'].str.lower())
    
    matches = 0
    for name in gt_names:
        if name in scraped_names:
            matches += 1
    
    print(f"Names that match: {matches}/{len(gt_names)}")


if __name__ == "__main__":
    import sys
    
    print("SHL Assessment Recommendation - URL Debug Tool")
    print("="*70)
    
    # Run analysis
    print("\n1. Analyzing URL mismatch...")
    analyze_url_mismatch()
    
    print("\n2. Attempting to create mapping...")
    mapping = create_url_mapping()
    
    print("\n3. Trying simple solution...")
    simple_solution_use_scraped_only()
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
The issue is that your ground truth URLs don't match the scraped URLs.

TWO OPTIONS:

Option 1: Fix the train.csv URLs (if you have the correct mapping)
  - Manually update train.csv with correct URLs from scraped data
  
Option 2: Evaluate by assessment NAME instead of URL (recommended)
  - Modify evaluation to match by fuzzy name comparison
  - This is more robust to URL changes

I'll create both solutions for you.
    """)