#!/usr/bin/env python3
"""
Unified PATSTAT ETL orchestrator - extracts core PET patent data and optional enrichment tables.

Usage Examples:

1. Basic extraction (core data only):
    python src/patstat/etl/run_patstat_export.py --resumable

2. Full extraction with enrichment (geography, institutions, citations, etc.):
    python src/patstat/etl/run_patstat_export.py --resumable --with-enrichment

3. Selective enrichment (only geography and institutions):
    python src/patstat/etl/run_patstat_export.py --resumable --with-enrichment --enrichment-priorities 1,2

4. Re-extract only enrichment tables (requires existing appln_core.parquet):
    python src/patstat/etl/extract_missing_tables.py --priorities 1,2,3

Enrichment Priorities:
    1 = Geographic data (90-95% coverage vs 18.5% in inventor_geo)
    2 = Institutional classification (universities, companies, government)
    3 = Firm linkage (harmonized applicant names)
    4 = Citation network (forward & backward citations)
    5 = Legal events (grants, renewals, abandonments)
    6 = ICT classification (technology codes)

Resumable Mode:
    --resumable processes data year-by-year and tracks completion state.
    If interrupted, rerun the same command to resume from where it left off.

Configuration can be overridden via environment variables:
    PATSTAT_DSN - ODBC DSN name (default: PATSTAT)
    PATSTAT_DB - Database name (default: patstat_2025_1)
    PATSTAT_OUT - Output directory (default: ./data/raw)
    PATSTAT_YEAR_FROM - Start year (default: 2010)
    PATSTAT_YEAR_TO - End year (default: 2025)
    PATSTAT_USE_ABSTRACTS - Include abstracts (default: 1, set to 0 to disable)
"""
import os
import sys
import argparse
import logging

# Add src directory to path (works from patstat dir or repo root)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(repo_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from patstat.etl import run_export, run_export_resumable


def parse_bool_env(value: str, default: bool = True) -> bool:
    """Parse boolean value from environment variable string.

    Handles common representations of False (0, 'False', 'false', etc.).
    Any other value is considered True.

    Args:
        value: Environment variable value as string
        default: Default value to return if value is None

    Returns:
        Boolean interpretation of the value

    Examples:
        parse_bool_env('1', True) -> True
        parse_bool_env('0', True) -> False
        parse_bool_env('false', True) -> False
        parse_bool_env(None, False) -> False
    """
    if value is None:
        return default
    return value not in ('0', 'False', 'false', 'none', 'None')


def parse_args():
    """Parse command-line arguments for PATSTAT ETL export.

    Returns:
        argparse.Namespace: Parsed arguments containing:
            - dsn: ODBC data source name
            - db: Database name
            - out: Output directory path
            - year_from: Start year for extraction
            - year_to: End year for extraction
            - use_abstracts: Whether to search abstracts (False if --no-abstracts)
            - dry_run: Print config only, don't run extraction
            - resumable: Enable year-by-year resumable mode
            - reset_state: Clear previous resumable state
            - with_enrichment: Extract enrichment tables after core extraction
            - enrichment_priorities: Comma-separated list of priorities (1-6)
    """
    parser = argparse.ArgumentParser(description='Run PATSTAT ETL export')

    # Database connection arguments
    parser.add_argument('--dsn', dest='dsn', help='ODBC DSN name (overrides PATSTAT_DSN)')
    parser.add_argument('--db', dest='db', help='Database name (overrides PATSTAT_DB)')

    # Output configuration
    parser.add_argument('--out', dest='out', help='Output directory (overrides PATSTAT_OUT)')

    # Time range configuration
    parser.add_argument('--year-from', dest='year_from', type=int, help='Start year')
    parser.add_argument('--year-to', dest='year_to', type=int, help='End year')

    # Extraction options
    parser.add_argument('--no-abstracts', dest='use_abstracts', action='store_false',
                       help='Disable abstracts search (only search titles, faster but lower recall)')

    # Execution mode
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                       help='Print configuration and exit without running')
    parser.add_argument('--resumable', dest='resumable', action='store_true',
                       help='Enable year-by-year resumable mode (recommended for unstable connections)')
    parser.add_argument('--reset-state', dest='reset_state', action='store_true',
                       help='Clear previous state and start fresh (use with --resumable)')

    # Enrichment options (optional second extraction phase)
    parser.add_argument('--with-enrichment', dest='with_enrichment', action='store_true',
                       help='Also extract enrichment tables after main extraction (geography, institutions, citations, etc.)')
    parser.add_argument('--enrichment-priorities', dest='enrichment_priorities',
                       help='Comma-separated priorities for enrichment (1-6): 1=Geography, 2=Institutions, 3=Firm linkage, 4=Citations, 5=Legal events, 6=ICT classification. Default: 1,2,3,4,5,6 (all)')

    return parser.parse_args()

# Configuration (can be overridden via environment variables or CLI args)
DSN_NAME = os.environ.get('PATSTAT_DSN', 'PATSTAT')
DB_NAME = os.environ.get('PATSTAT_DB', 'patstat_2025_1')
# Default output to src/patstat/data/raw
default_out = os.path.join(os.path.dirname(__file__), 'data', 'raw')
OUT_DIR = os.environ.get('PATSTAT_OUT', default_out)
YEAR_FROM = os.environ.get('PATSTAT_YEAR_FROM', '2010')
YEAR_TO = os.environ.get('PATSTAT_YEAR_TO', '2025')
USE_ABSTRACTS = parse_bool_env(os.environ.get('PATSTAT_USE_ABSTRACTS', None), default=True)


def resolved_config_from_env_and_cli():
    """Resolve configuration from environment variables and CLI arguments.

    Merges configuration from three sources (in priority order):
    1. Command-line arguments (highest priority)
    2. Environment variables (PATSTAT_*)
    3. Hard-coded defaults (lowest priority)

    Returns:
        dict: Resolved configuration with keys:
            - dsn (str): ODBC data source name
            - db (str): Database name
            - out (str): Output directory path
            - year_from (int): Start year for extraction
            - year_to (int): End year for extraction
            - use_abstracts (bool): Whether to search abstracts
            - dry_run (bool): Print config only, don't run
            - resumable (bool): Enable year-by-year resumable mode
            - reset_state (bool): Clear previous resumable state
            - with_enrichment (bool): Extract enrichment tables
            - enrichment_priorities (list[int] or None): List of priorities 1-6, or None for all

    Raises:
        SystemExit: If enrichment priorities are invalid or years are not integers
    """
    args = parse_args()

    # Resolve each config value with priority: CLI args > env vars > defaults
    dsn = args.dsn if args.dsn else DSN_NAME
    db = args.db if args.db else DB_NAME
    out = args.out if args.out else OUT_DIR
    year_from = args.year_from if args.year_from is not None else YEAR_FROM
    year_to = args.year_to if args.year_to is not None else YEAR_TO
    use_abstracts = args.use_abstracts if 'use_abstracts' in args.__dict__ and args.use_abstracts is not None else USE_ABSTRACTS

    # Boolean flags (always from CLI args)
    dry_run = args.dry_run
    resumable = args.resumable
    reset_state = args.reset_state
    with_enrichment = args.with_enrichment

    # Parse enrichment priorities from comma-separated string
    enrichment_priorities = None
    if args.enrichment_priorities:
        try:
            # Split on comma, strip whitespace, convert to int
            enrichment_priorities = [int(p.strip()) for p in args.enrichment_priorities.split(',')]

            # Validate: all priorities must be in range 1-6
            if not all(1 <= p <= 6 for p in enrichment_priorities):
                raise SystemExit('Enrichment priorities must be between 1 and 6')
        except ValueError:
            # If conversion to int fails, provide helpful error message
            raise SystemExit('Enrichment priorities must be comma-separated integers (e.g., "1,2,3")')

    # Normalize year types to integers
    try:
        year_from = int(year_from)
        year_to = int(year_to)
    except ValueError:
        raise SystemExit('PATSTAT_YEAR_FROM and PATSTAT_YEAR_TO must be integers')

    # Return all configuration as dictionary
    return dict(dsn=dsn, db=db, out=out, year_from=year_from, year_to=year_to,
                use_abstracts=use_abstracts, dry_run=dry_run, resumable=resumable,
                reset_state=reset_state, with_enrichment=with_enrichment,
                enrichment_priorities=enrichment_priorities)

if __name__ == '__main__':
    # Configure logging to output simple messages to console
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Step 1: Resolve final configuration from env vars and CLI args
    cfg = resolved_config_from_env_and_cli()

    # Extract configuration values for use in main execution
    DSN_NAME = cfg['dsn']
    DB_NAME = cfg['db']
    OUT_DIR = cfg['out']
    YEAR_FROM = cfg['year_from']
    YEAR_TO = cfg['year_to']
    USE_ABSTRACTS = cfg['use_abstracts']
    DRY_RUN = cfg['dry_run']
    RESUMABLE = cfg['resumable']
    RESET_STATE = cfg['reset_state']
    WITH_ENRICHMENT = cfg['with_enrichment']
    ENRICHMENT_PRIORITIES = cfg['enrichment_priorities']

    # Display configuration summary to user
    logging.info("%s", "=" * 60)
    logging.info("PATSTAT PET Patent Export")
    logging.info("%s", "=" * 60)
    logging.info("DSN:           %s", DSN_NAME)
    logging.info("Database:      %s", DB_NAME)
    logging.info("Output Dir:    %s", OUT_DIR)
    logging.info("Year Range:    %s-%s", YEAR_FROM, YEAR_TO)
    logging.info("Use Abstracts: %s", USE_ABSTRACTS)
    logging.info("Mode:          %s", "Resumable (year-by-year)" if RESUMABLE else "Standard (single batch)")

    # Show resumable-specific config
    if RESUMABLE and RESET_STATE:
        logging.info("Reset State:   %s", "Yes (will clear previous progress)")

    # Show enrichment config
    if WITH_ENRICHMENT:
        priorities_str = ','.join(map(str, ENRICHMENT_PRIORITIES)) if ENRICHMENT_PRIORITIES else 'All (1-6)'
        logging.info("Enrichment:    Enabled (Priorities: %s)", priorities_str)

    logging.info("%s", "=" * 60)

    # Handle dry-run mode: print config and exit without executing
    if DRY_RUN:
        logging.info('Dry-run: no export will be performed.')
        sys.exit(0)

    try:
        # ====================================================================
        # STEP 1: CORE EXTRACTION
        # Extract PET patents and core metadata tables
        # ====================================================================
        if RESUMABLE:
            # Resumable mode: Year-by-year extraction with state tracking
            # Can recover from failures by resuming from last completed year
            output_dir = run_export_resumable(
                dsn_name=DSN_NAME,
                db_name=DB_NAME,
                out_dir=OUT_DIR,
                year_from=YEAR_FROM,
                year_to=YEAR_TO,
                use_abstracts=USE_ABSTRACTS,
                reset_state=RESET_STATE
            )
        else:
            # Standard mode: Single-batch extraction (faster but no resumability)
            output_dir = run_export(
                dsn_name=DSN_NAME,
                db_name=DB_NAME,
                out_dir=OUT_DIR,
                year_from=YEAR_FROM,
                year_to=YEAR_TO,
                use_abstracts=USE_ABSTRACTS
            )

        # ====================================================================
        # STEP 2: ENRICHMENT EXTRACTION (Optional)
        # Extract additional tables for enhanced analysis
        # Requires appln_core.parquet from Step 1
        # ====================================================================
        if WITH_ENRICHMENT:
            print()
            print("=" * 60)
            print("Step 2: Extracting enrichment tables...")
            print("=" * 60)

            try:
                # Import enrichment extractor (lazy import to avoid circular dependencies)
                from patstat.etl.extract_missing_tables import MissingTablesExtractor

                # Build path to appln_core.parquet from Step 1
                appln_core_path = os.path.join(output_dir, 'appln_core.parquet')

                # Verify appln_core.parquet exists (required for enrichment)
                # This file contains the list of patent IDs to enrich
                if not os.path.exists(appln_core_path):
                    print(f"✗ Error: {appln_core_path} not found")
                    print("  Cannot run enrichment without appln_core.parquet")
                    sys.exit(1)

                # Initialize enrichment extractor with same database connection
                extractor = MissingTablesExtractor(
                    dsn=DSN_NAME,
                    db_name=DB_NAME,
                    out_dir=output_dir,  # Same output dir as core extraction
                    batch_size=5000       # Process 5000 IDs per SQL query
                )

                # Run enrichment extraction for selected priorities
                # None = all priorities (1-6), or list of specific priorities
                extractor.run(
                    source_file=appln_core_path,
                    priorities=ENRICHMENT_PRIORITIES,
                    dry_run=False
                )

                print()
                print("=" * 60)
                print("✓ Enrichment extraction completed successfully!")
                print("=" * 60)

            except ImportError as e:
                # Enrichment extractor module not found
                print(f"✗ Error importing MissingTablesExtractor: {e}")
                print("  Please ensure extract_missing_tables.py is available")
                sys.exit(1)
            except Exception as e:
                # Enrichment extraction failed, but core extraction succeeded
                # Allow user to retry enrichment separately
                print()
                print("=" * 60)
                print("✗ Enrichment extraction failed!")
                print("=" * 60)
                print(f"Error: {e}")
                print()
                print("Main extraction completed successfully, but enrichment failed.")
                print("You can retry enrichment separately using:")
                print(f"  python src/patstat/etl/extract_missing_tables.py --source {appln_core_path}")
                print("=" * 60)
                sys.exit(1)

        # ====================================================================
        # SUCCESS: Display summary of generated files
        # ====================================================================
        print()
        print("=" * 60)
        print("✓ Export completed successfully!")
        print("=" * 60)
        print(f"Output files saved to: {output_dir}")
        print()
        # List core files from Step 1 (always generated)
        print("Core files:")
        print("  • pet_ids_keyword_only.parquet    (keyword search, high recall)")
        print("  • pet_ids_intersection.parquet    (keyword + IPC, high precision)")
        print("  • pet_ids_ipc_only.parquet        (IPC classification only)")
        print("  • appln_core.parquet              (patent metadata)")
        print("  • title_abstract.parquet          (full text)")
        print("  • ipc.parquet, cpc.parquet        (classifications)")
        print("  • wipo.parquet                    (technology fields)")
        print("  • quality_epo.parquet             (quality metrics)")
        print("  • inventor_geo.parquet            (geography)")
        print("  • han_orbis.parquet               (firm linkage)")

        # List enrichment files from Step 2 (only if requested)
        if WITH_ENRICHMENT:
            print()
            print("Enrichment files:")
            # Show only the priorities that were actually extracted
            if not ENRICHMENT_PRIORITIES or 1 in ENRICHMENT_PRIORITIES:
                print("  • person_geography_complete.parquet  (enhanced geography 90-95% coverage)")
            if not ENRICHMENT_PRIORITIES or 2 in ENRICHMENT_PRIORITIES:
                print("  • person_institutions.parquet        (universities, companies, government)")
            if not ENRICHMENT_PRIORITIES or 3 in ENRICHMENT_PRIORITIES:
                print("  • han_applicants_complete.parquet    (harmonized firm names)")
            if not ENRICHMENT_PRIORITIES or 4 in ENRICHMENT_PRIORITIES:
                print("  • citations_complete.parquet         (forward & backward citations)")
            if not ENRICHMENT_PRIORITIES or 5 in ENRICHMENT_PRIORITIES:
                print("  • legal_events.parquet               (grants, renewals, abandonments)")
            if not ENRICHMENT_PRIORITIES or 6 in ENRICHMENT_PRIORITIES:
                print("  • ict_classification.parquet         (ICT technology codes)")

        print("=" * 60)

    except Exception as e:
        # ====================================================================
        # FAILURE: Core extraction failed
        # Provide helpful debugging guidance to user
        # ====================================================================
        print()
        print("=" * 60)
        print("✗ Export failed!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Please check:")
        print("  1. ODBC DSN is configured correctly")
        print("  2. Database name is correct")
        print("  3. You have network/VPN access to the database")
        print("  4. Output directory is writable")
        print("=" * 60)
        sys.exit(1)
