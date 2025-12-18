import pandas as pd
import json

def load_excel_data(excel_path='Gen_AI Dataset.xlsx'):
    """
    Load data from Excel file with train and test queries
    """
    print(f"Loading data from {excel_path}...")
    
    # Load train data
    train_df = pd.read_excel(excel_path, sheet_name='Train-Set')
    print(f"Train data: {len(train_df)} rows")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # Load test data
    test_df = pd.read_excel(excel_path, sheet_name='Test-Set')
    print(f"Test data: {len(test_df)} rows")
    print(f"Columns: {test_df.columns.tolist()}")
    
    # Save to CSV format for easier processing
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("Saved train.csv and test.csv")
    
    return train_df, test_df


def prepare_ground_truth(train_df):
    """
    Prepare ground truth in format: Query, Assessment_url
    """
    rows = []
    
    for _, row in train_df.iterrows():
        query = row['Query']
        # Assuming the Excel has columns with URLs for relevant assessments
        # Adjust column names based on actual structure
        for col in train_df.columns:
            if col.startswith('Assessment') or col.startswith('URL') or col.startswith('Relevant'):
                url = row[col]
                if pd.notna(url) and str(url).startswith('http'):
                    rows.append({
                        'Query': query,
                        'Assessment_url': url
                    })
    
    ground_truth_df = pd.DataFrame(rows)
    ground_truth_df.to_csv('data/ground_truth.csv', index=False)
    
    print(f"Created ground truth with {len(ground_truth_df)} rows")
    return ground_truth_df