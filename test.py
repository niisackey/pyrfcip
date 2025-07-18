# test.py
import pandas as pd
import time
from datetime import datetime
from typing import List, Union

# Replace with your actual import
# from your_module import get_price_data

# Mock implementation for testing (use your real function in production)
def get_price_data(
    year: Union[int, List[int]] = None,
    crop: Union[str, List[str]] = None,
    state: Union[str, List[str]] = None,
    max_retries: int = 1,
    request_delay: int = 1
) -> pd.DataFrame:
    """Mock implementation - replace with your actual function"""
    # In your real code, use the full implementation
    return pd.DataFrame({
        'year': [2023, 2023],
        'state': ['IA', 'IL'],
        'commodity': [crop, crop],
        'price': [5.75, 5.80]
    })

def test_crop_api(
    crops: List[str],
    years: List[int] = [2023, 2024, 2025],
    states: List[str] = ['IA', 'IL', 'NE', 'MN', 'US'],
    max_retries: int = 1,
    request_delay: int = 1
):
    """Test the USDA API for multiple crops, years, and states"""
    results = []
    
    for crop in crops:
        for year in years:
            for state in states:
                start_time = time.time()
                try:
                    logger.info(f"Testing: {crop} - {year} - {state}")
                    df = get_price_data(
                        year=year,
                        crop=crop,
                        state=state,
                        max_retries=max_retries,
                        request_delay=request_delay
                    )
                    
                    # Record results
                    elapsed = time.time() - start_time
                    record = {
                        'crop': crop,
                        'year': year,
                        'state': state,
                        'status': 'SUCCESS' if not df.empty else 'NO DATA',
                        'records': len(df),
                        'time_sec': round(elapsed, 2),
                        'min_price': df['price'].min() if not df.empty else None,
                        'max_price': df['price'].max() if not df.empty else None,
                        'sample_data': df.head(1).to_dict('records') if not df.empty else None
                    }
                    results.append(record)
                    
                    logger.info(f"Result: {record['status']} - {record['records']} records")
                    
                except Exception as e:
                    elapsed = time.time() - start_time
                    results.append({
                        'crop': crop,
                        'year': year,
                        'state': state,
                        'status': 'ERROR',
                        'error': str(e),
                        'time_sec': round(elapsed, 2)
                    })
                    logger.error(f"Error for {crop}-{year}-{state}: {str(e)}")
                
                time.sleep(request_delay)  # Be kind to the API
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"usda_api_test_{timestamp}.csv"
    results_df.to_csv(filename, index=False)
    
    # Print summary
    print("\nTest Summary:")
    print(f"Total tests: {len(results_df)}")
    print(f"Successful: {len(results_df[results_df['status'] == 'SUCCESS'])}")
    print(f"No Data: {len(results_df[results_df['status'] == 'NO DATA'])}")
    print(f"Errors: {len(results_df[results_df['status'] == 'ERROR'])}")
    
    # Print problematic cases
    if not results_df[results_df['status'] == 'SUCCESS'].empty:
        print("\nSuccessful crops:")
        print(results_df[results_df['status'] == 'SUCCESS']['crop'].unique())
    
    if not results_df[results_df['status'] == 'NO DATA'].empty:
        print("\nCrops with no data:")
        print(results_df[results_df['status'] == 'NO DATA'][['crop', 'year', 'state']].drop_duplicates())
    
    if not results_df[results_df['status'] == 'ERROR'].empty:
        print("\nCrops with errors:")
        print(results_df[results_df['status'] == 'ERROR'][['crop', 'year', 'state', 'error']].drop_duplicates())
    
    print(f"\nFull results saved to {filename}")
    return results_df

if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('USDA_API_Test')
    
    # List of crops to test
    crops_to_test = [
        'CORN', 'SOYBEANS', 'WHEAT', 'COTTON', 'RICE',
        'BARLEY', 'OATS', 'SORGHUM', 'PEANUTS', 'BEANS',
        'SUGARBEETS', 'SUNFLOWER', 'CANOLA', 'CATTLE',
        'HOGS', 'MILK', 'EGGS', 'CHICKENS', 'TURKEYS'
    ]
    
    # Years to test (adjust as needed)
    test_years = [2020, 2021, 2022, 2023, 2024, 2025]
    
    # States to test (use 'US' for national data)
    test_states = ['IA', 'IL', 'NE', 'MN', 'US']  # Add more as needed
    
    # Run the test
    results = test_crop_api(
        crops=crops_to_test,
        years=test_years,
        states=test_states,
        max_retries=1,  # Increase if needed
        request_delay=1  # Seconds between requests
    )