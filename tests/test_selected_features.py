"""
Test script for extracting and saving multiple selected features to a test table.

Usage:
    python tests/test_selected_features.py --features gold,oil,dxy --start 2024-01-01
    python tests/test_selected_features.py --features market --start 2024-01-01 --end 2024-12-31
    python tests/test_selected_features.py --features gold,th_macro --start 2024-01-01
"""

import argparse
from datetime import datetime

import project_paths  # noqa: F401
from apis.extract import extract_selected_features, FEATURE_GROUPS
from database.save_db import create_test_table, save_fx_features
from features.cleaning import handle_missing_values


def test_selected_features(features: list[str], start_date: str, end_date: str = None, table_name: str = None):
    """Test extraction of multiple selected features."""
    
    end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")
    
    if not table_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"test_custom_{timestamp}"
    
    print(f"\n🧪 Testing features: {', '.join(features)}")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📊 Table name: {table_name}\n")
    
    # Extract data
    print("1️⃣ Extracting data...")
    df = extract_selected_features(start_date, end_date, features=features)
    
    if df.empty:
        print("❌ No data extracted. Check your feature names and date range.")
        return
    
    print(f"   Extracted {len(df)} rows with columns: {list(df.columns)}\n")
    
    # Clean data
    print("2️⃣ Cleaning data...")
    df = handle_missing_values(df)
    print(f"   After cleaning: {len(df)} rows\n")
    
    # Create test table
    print("3️⃣ Creating test table...")
    columns = ["date"] + list(df.columns)
    create_test_table(table_name, columns)
    
    # Save data
    print("4️⃣ Saving data to test table...")
    save_fx_features(df, table_name=table_name)
    
    # Summary
    print("\n📈 Summary Statistics:")
    print(df.describe())
    
    print(f"\n✅ Test complete! Data saved to table '{table_name}'")
    print("\nTo drop the test table, run:")
    print(f"   python tests/cleanup.py --tables {table_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Test multiple feature extraction",
        epilog=f"Available feature groups: {', '.join(FEATURE_GROUPS.keys())}"
    )
    parser.add_argument(
        "--features", 
        required=True, 
        help="Comma-separated list of features or groups (e.g., 'gold,oil' or 'market' or 'gold,th_macro')"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--table", help="Custom table name (optional)")
    
    args = parser.parse_args()
    
    features = [f.strip() for f in args.features.split(",")]
    
    test_selected_features(features, args.start, args.end, args.table)


if __name__ == "__main__":
    main()
