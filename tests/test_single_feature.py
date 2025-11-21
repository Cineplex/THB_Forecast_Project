"""
Test script for extracting and saving a single feature to a test table.

Usage:
    python tests/test_single_feature.py --feature gold --start 2024-01-01 --end 2024-12-31
    python tests/test_single_feature.py --feature usd_thb --start 2024-01-01
"""

import argparse
from datetime import datetime

import project_paths  # noqa: F401
from apis.extract import extract_selected_features
from database.save_db import create_test_table, save_fx_features, drop_table
from features.cleaning import handle_missing_values


def test_single_feature(feature: str, start_date: str, end_date: str = None):
    """Test extraction of a single feature."""
    
    end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")
    table_name = f"test_{feature}"
    
    print(f"\n🧪 Testing feature: {feature}")
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"📊 Table name: {table_name}\n")
    
    # Extract data
    print("1️⃣ Extracting data...")
    df = extract_selected_features(start_date, end_date, features=[feature])
    
    if df.empty:
        print("❌ No data extracted. Check your feature name and date range.")
        return
    
    print(f"   Extracted {len(df)} rows\n")
    
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
    parser = argparse.ArgumentParser(description="Test single feature extraction")
    parser.add_argument("--feature", required=True, help="Feature name (e.g., gold, oil, usd_thb)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), defaults to today")
    
    args = parser.parse_args()
    
    test_single_feature(args.feature, args.start, args.end)


if __name__ == "__main__":
    main()
