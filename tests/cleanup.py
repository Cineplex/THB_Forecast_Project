"""
Cleanup utility for dropping test tables.

Usage:
    python tests/cleanup.py --list                    # List all test tables
    python tests/cleanup.py --tables test_gold        # Drop specific table
    python tests/cleanup.py --tables test_gold,test_oil  # Drop multiple tables
    python tests/cleanup.py --all                     # Drop all test tables
    python tests/cleanup.py --all --no-confirm        # Drop all without confirmation
"""

import argparse

import project_paths  # noqa: F401
from database.save_db import drop_table, list_test_tables


def cleanup_tables(tables: list[str] = None, drop_all: bool = False, no_confirm: bool = False):
    """Drop test tables."""
    
    if drop_all:
        test_tables = list_test_tables()
        if not test_tables:
            print("ℹ️  No test tables found.")
            return
        
        print(f"📋 Found {len(test_tables)} test table(s):")
        for table in test_tables:
            print(f"  - {table}")
        
        print()
        
        if not no_confirm:
            response = input(f"⚠️  Drop all {len(test_tables)} test tables? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("❌ Cancelled.")
                return
        
        for table in test_tables:
            drop_table(table, confirm=False)
        
        print(f"\n✅ Dropped {len(test_tables)} test table(s).")
    
    elif tables:
        for table in tables:
            drop_table(table, confirm=not no_confirm)
    
    else:
        print("❌ Please specify --tables or --all")


def list_tables():
    """List all test tables."""
    test_tables = list_test_tables()
    
    if not test_tables:
        print("ℹ️  No test tables found.")
        return
    
    print(f"📋 Found {len(test_tables)} test table(s):\n")
    for table in test_tables:
        print(f"  - {table}")
    
    print(f"\nTo drop specific tables:")
    print(f"  python tests/cleanup.py --tables {test_tables[0]}")
    print(f"\nTo drop all test tables:")
    print(f"  python tests/cleanup.py --all")


def main():
    parser = argparse.ArgumentParser(description="Cleanup test tables")
    parser.add_argument("--list", action="store_true", help="List all test tables")
    parser.add_argument("--tables", help="Comma-separated list of table names to drop")
    parser.add_argument("--all", action="store_true", help="Drop all test tables")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if args.list:
        list_tables()
    else:
        tables = [t.strip() for t in args.tables.split(",")] if args.tables else None
        cleanup_tables(tables, args.all, args.no_confirm)


if __name__ == "__main__":
    main()
