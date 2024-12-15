import boto3
import pandas as pd
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict
import time
from tqdm import tqdm
import math
import os
from dotenv import load_dotenv

# At the top of your file, after imports
load_dotenv()

class ParallelDynamoProcessor:
    def __init__(self, table_name: str, segment_count: int = 8):
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=os.getenv('AWS_DEFAULT_REGION'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.table = self.dynamodb.Table(table_name)
        self.segment_count = segment_count

    def parallel_scan(self, segment: int, total_segments: int) -> List[Dict]:
        """Scan a segment of the DynamoDB table"""
        items = []
        scan_kwargs = {
            'Segment': segment,
            'TotalSegments': total_segments,
            'Limit': 1000  # Added a reasonable limit per request
        }
        
        try:
            done = False
            start_key = None
            target_items = 10000 // total_segments
            
            while not done and len(items) < target_items:
                if start_key:
                    scan_kwargs['ExclusiveStartKey'] = start_key
                response = self.table.scan(**scan_kwargs)
                
                # Convert DynamoDB types to Python types
                for item in response.get('Items', []):
                    # Ensure card4 and card6 are loaded as strings
                    if 'card4' in item:
                        item['card4'] = str(item['card4'])  # Convert to string
                    if 'card6' in item:
                        item['card6'] = str(item['card6'])  # Convert to string
                    items.append(item)
                    
                start_key = response.get('LastEvaluatedKey', None)
                done = start_key is None
                
        except Exception as e:
            print(f"Error in segment {segment}: {str(e)}")
            
        return items[:target_items]

    def fetch_data_parallel(self) -> pd.DataFrame:
        """Fetch data using parallel processing"""
        print(f"Starting parallel scan with {self.segment_count} segments...")
        start_time = time.time()
        
        # Using ThreadPoolExecutor for I/O bound operations (DynamoDB scanning)
        with ThreadPoolExecutor(max_workers=self.segment_count) as executor:
            futures = [
                executor.submit(self.parallel_scan, segment, self.segment_count)
                for segment in range(self.segment_count)
            ]
            
            # Collect results with progress bar
            all_items = []
            for future in tqdm(futures, desc="Scanning segments"):
                all_items.extend(future.result())
        
        end_time = time.time()
        print(f"Scan completed in {end_time - start_time:.2f} seconds")
        print(f"Total records fetched: {len(all_items)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_items)
        return df

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of data"""
    numeric_columns = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5']
    for col in numeric_columns:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    
    # Ensure card4 and card6 are treated as categories
    categorical_columns = ['card4', 'card6']
    for col in categorical_columns:
        if col in chunk.columns:
            chunk[col] = chunk[col].astype('category')
    
    return chunk

def parallel_process_data(df: pd.DataFrame, num_processes: int = None) -> pd.DataFrame:
    """Process the DataFrame using parallel processing"""
    if df.empty:
        print("Warning: Empty DataFrame received, returning as is.")
        return df
        
    if num_processes is None:
        num_processes = mp.cpu_count()
    print("Number of process: ",num_processes)
    # Split DataFrame into chunks, ensure at least 1 row per chunk
    chunk_size = max(1, math.ceil(len(df) / num_processes))
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        processed_chunks = list(tqdm(
            executor.map(process_chunk, chunks),
            total=len(chunks),
            desc="Processing chunks"
        ))
    
    # Combine processed chunks
    return pd.concat(processed_chunks, ignore_index=True)

def main():
    # Initialize parallel processor
    processor = ParallelDynamoProcessor('Fraud_Detection', segment_count=8)
    
    # Fetch data in parallel
    print("Fetching data from DynamoDB...")
    df = processor.fetch_data_parallel()
    
    # Process data in parallel
    print("\nProcessing data in parallel...")
    df_processed = parallel_process_data(df)
    
    print("\nData ready for analysis!")
    print(f"Final dataset shape: {df_processed.shape}")
    
    return df_processed

# Run the analysis functions after getting the processed data
def run_analysis(df):
    # Analysis functions from previous response...
    # (Include all the analysis functions from the previous code here)
    pass

if __name__ == "__main__":
    # Get processed data
    processed_df = main()
    print(processed_df)
    # Run analysis
    run_analysis(processed_df)