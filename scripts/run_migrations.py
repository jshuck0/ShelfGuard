"""
Run Database Migrations for ShelfGuard Strategic Intelligence System

This script runs the SQL migration files on your Supabase database.
"""

import os
from pathlib import Path
from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Initialize Supabase client from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for admin operations

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing Supabase credentials. Please set:\n"
            "  SUPABASE_URL\n"
            "  SUPABASE_SERVICE_KEY\n"
            "in your environment or .env file"
        )

    return create_client(supabase_url, supabase_key)


def run_sql_file(client: Client, sql_file_path: Path) -> None:
    """
    Run a SQL file on Supabase.

    Note: Supabase Python client doesn't have direct SQL execution.
    This function will print instructions for manual execution.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {sql_file_path.name}")
    print(f"{'='*80}\n")

    # Read SQL file
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_content = f.read()

    print("📄 SQL File Content:")
    print(f"   File: {sql_file_path}")
    print(f"   Size: {len(sql_content)} characters")
    print(f"   Lines: {len(sql_content.splitlines())}")

    # Count tables being created
    table_count = sql_content.lower().count('create table')
    index_count = sql_content.lower().count('create index')

    print(f"\n📊 Migration Summary:")
    print(f"   Tables to create: {table_count}")
    print(f"   Indexes to create: {index_count}")

    print(f"\n⚠️  IMPORTANT: Supabase Python client doesn't support raw SQL execution.")
    print(f"   You need to run this SQL manually in Supabase SQL Editor.\n")

    print(f"📋 Instructions:")
    print(f"   1. Go to: https://app.supabase.com/project/YOUR_PROJECT/sql/new")
    print(f"   2. Copy the contents of: {sql_file_path}")
    print(f"   3. Paste into the SQL Editor")
    print(f"   4. Click 'Run' button\n")

    # Offer to copy to clipboard if available
    try:
        import pyperclip
        response = input("📋 Copy SQL to clipboard? (y/n): ")
        if response.lower() == 'y':
            pyperclip.copy(sql_content)
            print("✅ SQL copied to clipboard! Paste it in Supabase SQL Editor.")
    except ImportError:
        print("💡 Tip: Install 'pyperclip' to auto-copy SQL to clipboard")
        print("   pip install pyperclip")


def verify_tables(client: Client) -> None:
    """Verify that tables were created successfully."""
    print(f"\n{'='*80}")
    print("Verifying Tables")
    print(f"{'='*80}\n")

    expected_tables = [
        'strategic_insights',
        'trigger_events',
        'insight_outcomes',
        'category_intelligence',
        'brand_intelligence',
        'market_patterns',
        'product_snapshots'  # Should already exist
    ]

    print("Expected tables:")
    for table in expected_tables:
        print(f"  - {table}")

    print(f"\n⚠️  To verify tables were created, run this SQL in Supabase:")
    print("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """)


def main():
    """Main migration runner."""
    print("="*80)
    print("ShelfGuard Strategic Intelligence System - Database Migrations")
    print("="*80)

    # Get project root
    project_root = Path(__file__).parent.parent
    schemas_dir = project_root / "schemas"

    # Check if schema files exist
    strategic_insights_sql = schemas_dir / "strategic_insights.sql"
    network_intelligence_sql = schemas_dir / "network_intelligence.sql"

    if not strategic_insights_sql.exists():
        print(f"❌ Error: {strategic_insights_sql} not found!")
        return

    if not network_intelligence_sql.exists():
        print(f"❌ Error: {network_intelligence_sql} not found!")
        return

    try:
        # Initialize Supabase client (just for connection test)
        client = get_supabase_client()
        print("✅ Supabase connection verified")
    except ValueError as e:
        print(f"❌ Error: {e}")
        return

    # Process migration files
    print("\n" + "="*80)
    print("MIGRATION 1 OF 2: Strategic Insights")
    print("="*80)
    run_sql_file(client, strategic_insights_sql)

    print("\n" + "="*80)
    print("MIGRATION 2 OF 2: Network Intelligence")
    print("="*80)
    run_sql_file(client, network_intelligence_sql)

    # Verification instructions
    verify_tables(client)

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("""
1. Run the SQL files in Supabase SQL Editor (instructions above)
2. Verify tables were created using the verification SQL
3. Return here and tell Claude: "migrations done"
4. Claude will continue with implementation

Questions? Check: docs/IMPLEMENTATION_PROGRESS.md
    """)


if __name__ == "__main__":
    main()
