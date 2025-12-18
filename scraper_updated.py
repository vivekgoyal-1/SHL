import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from typing import List, Dict
import time

class SHLScraperUpdated:
    def __init__(self):
        self.base_url = "https://www.shl.com/products/product-catalog/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def scrape_individual_tests(self) -> List[Dict]:
        """Scrape individual test solutions (not pre-packaged)"""
        print("Scraping SHL Individual Test Solutions...")
        assessments = []
        
        # The catalog has pagination - need to get all pages
        page = 0
        start = 0
        
        while True:
            url = f"{self.base_url}?start={start}&type=1"  # type=1 for Individual Tests
            print(f"Fetching page {page + 1} (start={start})...")
            
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the table with individual tests
                table = soup.find('table')
                if not table:
                    break
                
                rows = table.find_all('tr')[1:]  # Skip header row
                
                if not rows:
                    break
                
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) >= 4:
                            # Extract assessment details
                            name_cell = cols[0]
                            test_type_cell = cols[3]
                            
                            link = name_cell.find('a')
                            if link:
                                name = link.get_text(strip=True)
                                relative_url = link.get('href', '')
                                
                                # Build full URL
                                if relative_url.startswith('http'):
                                    full_url = relative_url
                                else:
                                    full_url = f"https://www.shl.com{relative_url}"
                                
                                # Extract test types
                                test_types = test_type_cell.get_text(strip=True)
                                
                                assessment = {
                                    'name': name,
                                    'url': full_url,
                                    'test_types': test_types,
                                    'remote_testing': 'Yes' if cols[1].get_text(strip=True) else 'No',
                                    'adaptive': 'Yes' if cols[2].get_text(strip=True) else 'No'
                                }
                                
                                assessments.append(assessment)
                                
                    except Exception as e:
                        print(f"Error parsing row: {e}")
                        continue
                
                print(f"Found {len(assessments)} assessments so far...")
                
                # Check if there's a next page
                pagination = soup.find('ul', class_=['pagination', 'pager'])
                if pagination:
                    next_link = pagination.find('a', text='Next')
                    if next_link:
                        page += 1
                        start += 12  # Each page shows 12 items
                        time.sleep(1)  # Be nice to the server
                    else:
                        break
                else:
                    break
                    
            except Exception as e:
                print(f"Error fetching page: {e}")
                break
        
        print(f"\nTotal assessments scraped: {len(assessments)}")
        
        # Save to JSON
        with open('data/assessments.json', 'w', encoding='utf-8') as f:
            json.dump(assessments, f, indent=2, ensure_ascii=False)
        
        # Save to CSV
        df = pd.DataFrame(assessments)
        df.to_csv('data/assessments.csv', index=False)
        
        return assessments
    
    def scrape_with_details(self) -> List[Dict]:
        """
        Enhanced scraper that also fetches details from each assessment page
        This provides richer descriptions for better recommendations
        """
        base_assessments = self.scrape_individual_tests()
        
        print("\nEnhancing assessments with detailed descriptions...")
        
        for i, assessment in enumerate(base_assessments[:50]):  # Limit to first 50 for speed
            try:
                print(f"Fetching details {i+1}/50: {assessment['name'][:40]}...")
                response = requests.get(assessment['url'], headers=self.headers, timeout=15)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract description
                description = ""
                desc_elem = soup.find('div', class_=['description', 'content', 'summary'])
                if desc_elem:
                    description = desc_elem.get_text(strip=True)
                
                assessment['description'] = description[:500]  # Limit length
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                assessment['description'] = ""
                print(f"  Error: {e}")
        
        # Save enhanced data
        with open('data/assessments_enhanced.json', 'w', encoding='utf-8') as f:
            json.dump(base_assessments, f, indent=2, ensure_ascii=False)
        
        return base_assessments