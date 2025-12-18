

import pandas as pd
import json
import os

def analyze_excel_structure(excel_path='Gen_AI Dataset.xlsx'):
    """
    Analyze the structure of the Excel file
    """
    print("="*70)
    print("Analyzing Excel File Structure")
    print("="*70)
    
    # Load Excel file
    xl_file = pd.ExcelFile(excel_path)
    
    print(f"\nSheet names: {xl_file.sheet_names}")
    
    for sheet_name in xl_file.sheet_names:
        print(f"\n{'-'*70}")
        print(f"Sheet: {sheet_name}")
        print(f"{'-'*70}")
        
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        print(f"Shape: {df.shape} (rows x columns)")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string())
        print(f"\nColumn data types:")
        print(df.dtypes)


def process_train_data(excel_path='Gen_AI Dataset.xlsx', sheet_name='Train-Set'):
    """
    Process training data into proper format
    Expected format: Query | Assessment URLs (one or more columns)
    """
    print("\n" + "="*70)
    print("Processing Training Data")
    print("="*70)
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    print(f"Loaded {len(df)} queries")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create ground truth format: Query, Assessment_url
    rows = []
    
    for idx, row in df.iterrows():
        query = row.iloc[0]  # First column should be the query
        
        # Look for URL columns (usually subsequent columns)
        for col_idx in range(1, len(row)):
            value = row.iloc[col_idx]
            
            # Check if it's a URL
            if pd.notna(value):
                value_str = str(value).strip()
                if value_str.startswith('http'):
                    rows.append({
                        'Query': query,
                        'Assessment_url': value_str
                    })
    
    # Save to CSV
    train_df = pd.DataFrame(rows)
    
    if len(train_df) == 0:
        print("\nWARNING: No URLs found in expected format!")
        print("Please check the Excel structure.")
        print("\nExpected structure:")
        print("  Column 0: Query text")
        print("  Column 1+: Assessment URLs")
        return None
    
    os.makedirs('data', exist_ok=True)
    train_df.to_csv('data/train.csv', index=False)
    
    print(f"\n✓ Created train.csv with {len(train_df)} rows")
    print(f"  Unique queries: {train_df['Query'].nunique()}")
    
    # Show summary
    query_counts = train_df.groupby('Query').size()
    print(f"  Assessments per query: {query_counts.min()} to {query_counts.max()}")
    
    print("\nSample:")
    print(train_df.head().to_string(index=False))
    
    return train_df


def process_test_data(excel_path='Gen_AI Dataset.xlsx', sheet_name='Test-Set'):
    """
    Process test data - just queries, no ground truth
    """
    print("\n" + "="*70)
    print("Processing Test Data")
    print("="*70)
    
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    print(f"Loaded {len(df)} test queries")
    
    # Extract just the queries (first column)
    test_queries = pd.DataFrame({
        'Query': df.iloc[:, 0]
    })
    
    os.makedirs('data', exist_ok=True)
    test_queries.to_csv('data/test.csv', index=False)
    
    print(f"\n✓ Created test.csv with {len(test_queries)} queries")
    
    print("\nTest queries:")
    for i, query in enumerate(test_queries['Query'], 1):
        print(f"{i}. {query}")
    
    return test_queries


def create_sample_data():
    """
    Create sample data if Excel file is not available
    """
    print("\n" + "="*70)
    print("Creating Sample Data")
    print("="*70)
    
    # Sample training data
    train_data = [
        {
            'Query': 'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            'Assessment_url': 'https://www.shl.com/products/product-catalog/view/java-programming-test/'
        },
        {
            'Query': 'I am hiring for Java developers who can also collaborate effectively with my business teams.',
            'Assessment_url': 'https://www.shl.com/products/product-catalog/view/teamwork-sjt/'
        },
        {
            'Query': 'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Assessment_url': 'https://www.shl.com/products/product-catalog/view/python-programming/'
        },
        {
            'Query': 'Looking to hire mid-level professionals who are proficient in Python, SQL and JavaScript.',
            'Assessment_url': 'https://www.shl.com/products/product-catalog/view/sql-fundamentals/'
        },
        {
            'Query': 'Need cognitive and personality tests for analyst position screening.',
            'Assessment_url': 'https://www.shl.com/products/product-catalog/view/verify-g-plus/'
        },
    ]
    
    # Sample test data
    test_data = [
        {'Query': 'Hiring software engineer with leadership skills'},
        {'Query': 'Looking for customer service representative'},
        {'Query': 'Need assessments for data analyst position'},
    ]
    
    os.makedirs('data', exist_ok=True)
    
    # Save train data
    train_df = pd.DataFrame(train_data)
    train_df.to_csv('data/train.csv', index=False)
    print(f"✓ Created sample train.csv with {len(train_df)} rows")
    
    # Save test data
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('data/test.csv', index=False)
    print(f"✓ Created sample test.csv with {len(test_df)} queries")


def validate_data_format():
    """
    Validate that train.csv and test.csv are in correct format
    """
    print("\n" + "="*70)
    print("Validating Data Format")
    print("="*70)
    
    errors = []
    
    # Check train.csv
    try:
        train_df = pd.read_csv('data/train.csv')
        
        print("\nTrain data:")
        print(f"  Shape: {train_df.shape}")
        print(f"  Columns: {train_df.columns.tolist()}")
        
        required_cols = ['Query', 'Assessment_url']
        if list(train_df.columns) != required_cols:
            errors.append(f"Train columns must be {required_cols}, got {list(train_df.columns)}")
        
        # Check for nulls
        if train_df.isnull().any().any():
            errors.append("Train data contains null values")
        
        # Check URLs
        urls = train_df['Assessment_url'].tolist()
        non_urls = [url for url in urls if not str(url).startswith('http')]
        if non_urls:
            errors.append(f"Train data contains {len(non_urls)} non-URL values")
        
        print(f"  ✓ Train data looks good")
        
    except Exception as e:
        errors.append(f"Error loading train.csv: {e}")
    
    # Check test.csv
    try:
        test_df = pd.read_csv('data/test.csv')
        
        print("\nTest data:")
        print(f"  Shape: {test_df.shape}")
        print(f"  Columns: {test_df.columns.tolist()}")
        
        if 'Query' not in test_df.columns:
            errors.append("Test data must have 'Query' column")
        
        print(f"  ✓ Test data looks good")
        
    except Exception as e:
        errors.append(f"Error loading test.csv: {e}")
    
    if errors:
        print("\n Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n All data files are valid!")
        return True


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("SHL Assessment Data Handler")
    print("="*70)
    
    excel_file = 'Gen_AI Dataset.xlsx'
    
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"\n Excel file '{excel_file}' not found!")
        print("\nOptions:")
        print("1. Place 'Gen_AI Dataset.xlsx' in the current directory")
        print("2. Create sample data for testing")
        
        choice = input("\nCreate sample data? [y/N]: ")
        if choice.lower() == 'y':
            create_sample_data()
            validate_data_format()
        sys.exit(1)
    
    # Analyze structure
    print("\n" + "="*70)
    print("Step 1: Analyzing Excel Structure")
    print("="*70)
    analyze_excel_structure(excel_file)
    
    # Process train data
    print("\n" + "="*70)
    print("Step 2: Processing Train Data")
    print("="*70)
    train_df = process_train_data(excel_file)
    
    if train_df is None:
        print("\n Failed to process train data!")
        print("Please check the Excel file format.")
        
        # Show expected format
        print("\nExpected Excel structure:")
        print("="*70)
        print("Sheet: Train")
        print("-"*70)
        print("| Query                              | Assessment_URL_1      | Assessment_URL_2      |")
        print("|------------------------------------|-----------------------|-----------------------|")
        print("| Hiring Java developer with...     | https://shl.com/...   | https://shl.com/...   |")
        print("| Looking for data analyst who...   | https://shl.com/...   | https://shl.com/...   |")
        print("-"*70)
        
        sys.exit(1)
    
    # Process test data
    print("\n" + "="*70)
    print("Step 3: Processing Test Data")
    print("="*70)
    test_df = process_test_data(excel_file)
    
    # Validate
    print("\n" + "="*70)
    print("Step 4: Validating Data")
    print("="*70)
    is_valid = validate_data_format()
    
    if is_valid:
        print("\n" + "="*70)
        print(" Data processing complete!")
        print("="*70)
        print("\nGenerated files:")
        print("  - data/train.csv (ground truth)")
        print("  - data/test.csv (queries only)")
        print("\nYou can now run:")
        print("  1. python scraper_updated.py (scrape SHL catalog)")
        print("  2. python api.py (start API)")
        print("  3. python generate_predictions.py (generate predictions)")
    else:
        print("\n Data validation failed. Please fix errors above.")


# ==================== Additional utility functions ====================

def convert_to_submission_format(predictions_dict: dict, output_file='predictions.csv'):
    """
    Convert predictions dictionary to submission CSV format
    
    Args:
        predictions_dict: {query: [url1, url2, ...]}
        output_file: Output CSV filename
    """
    rows = []
    
    for query, urls in predictions_dict.items():
        for url in urls:
            rows.append({
                'Query': query,
                'Assessment_url': url
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    
    print(f"Saved {len(rows)} predictions to {output_file}")
    return df


def load_predictions_from_json(json_file='predictions.json'):
    """
    Load predictions from JSON and convert to submission format
    """
    with open(json_file, 'r') as f:
        predictions = json.load(f)
    
    return convert_to_submission_format(predictions)