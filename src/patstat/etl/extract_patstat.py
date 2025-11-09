"""PATSTAT ETL implementation with year-by-year processing and fault tolerance.

This module provides:
1. Year-by-year extraction (can process all years or resume from failures)
2. State tracking for completed years
3. Automatic resumption from the last incomplete year
4. Both single-year and multi-year export functions
"""
import pyodbc
import pandas as pd
import os
import textwrap
import itertools
import logging
from typing import Optional

from .state import ETLState
from .keywords import load_level1_keywords

# Module-level logger configuration
# Sets up logging to track ETL progress and errors
logger = logging.getLogger(__name__)
if not logger.handlers:
    try:
        # Create log file in the ETL directory
        etl_dir = os.path.dirname(__file__)
        log_path = os.path.join(etl_dir, 'collection_export.log')
        # Configure file handler with UTF-8 encoding for international characters
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setLevel(logging.INFO)
        # Format: timestamp, level, message
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # Silently fail if logging setup fails (e.g., permission issues)
        pass
logger.setLevel(logging.INFO)


def ensure_out_dir(path: str) -> str:
    """Ensure output directory exists, create if needed.

    Args:
        path: Directory path to create

    Returns:
        The same directory path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def run_export_single_year(
    conn: pyodbc.Connection,
    db_name: str,
    year: int,
    out_dir: str,
    level1_terms: list,
    use_abstracts: bool = True
) -> int:
    """Export PET patents for a single priority year from PATSTAT.

    This function implements the core extraction logic:
    1. Identifies PET patents using both keyword search and classification codes
    2. Exports three patent identification strategies (keyword-only, intersection, classification-only)
    3. Extracts 8 related data tables for identified patents
    4. Appends results to consolidated parquet files (enables year-by-year resumable extraction)

    The extraction uses a dual-approach strategy:
    - Keywords: Text search in titles/abstracts (high recall)
    - Classifications: IPC/CPC codes (high precision)
    - Intersection: Both keywords AND codes (highest confidence)

    Args:
        conn: Active pyodbc database connection to PATSTAT
        db_name: Database name in PATSTAT (e.g., 'patstat_2025_1')
        year: Priority year to extract (filters patents by PRIO_YEAR)
        out_dir: Output directory where parquet files will be saved/appended
        level1_terms: List of PET keyword terms to search for (loaded from gold CSV)
        use_abstracts: If True, search both titles and abstracts; if False, titles only
                      (abstracts increase recall but slow down extraction significantly)

    Returns:
        int: Number of unique patents extracted for this year

    Side Effects:
        - Creates/updates 11 parquet files in out_dir by appending new data:
          * pet_ids_keyword_only.parquet
          * pet_ids_intersection.parquet
          * pet_ids_classification_only.parquet
          * appln_core.parquet
          * title_abstract.parquet
          * ipc.parquet
          * cpc.parquet
          * wipo.parquet
          * quality_epo.parquet
          * inventor_geo.parquet
          * han_orbis.parquet
        - Writes progress to ETL log file

    Note:
        Uses temporary tables in SQL Server tempdb for efficient filtering.
        All temp tables are automatically cleaned up when connection closes.
    """
    # Create database cursor for executing queries
    cur = conn.cursor()

    def exec_sql(sql):
        """Execute SQL statement without returning results.

        Args:
            sql: SQL statement to execute
        """
        cur.execute(sql)

    def q(sql):
        """Remove leading indentation from SQL string for readability.

        Args:
            sql: SQL string with leading indentation

        Returns:
            SQL string with indentation removed
        """
        return textwrap.dedent(sql)

    logger.info('Processing year %d', year)

    # Create year-specific temp table for applications
    # This filters all patent applications by priority year to limit search space
    # Priority year is used as the anchor date for patent families
    exec_sql(q(f"""
        -- Switch to the PATSTAT database
        USE {db_name};

        -- Create temp table containing all patent applications for this specific year
        -- Drop if exists to ensure clean state
        IF OBJECT_ID('tempdb..#appln_year','U') IS NOT NULL DROP TABLE #appln_year;
        -- Get all application IDs that have this priority year
        -- JOIN with SPRING25_PRIO_APP to filter by priority year
        SELECT DISTINCT a.appln_id
        INTO #appln_year
        FROM SPRING25_APPLN a
        JOIN SPRING25_PRIO_APP pr ON pr.appln_id = a.appln_id
        WHERE pr.prio_year = {year};

        -- Create index on appln_id for faster joins in subsequent queries
        CREATE INDEX IX_year_appln ON #appln_year(appln_id);
    """))

    # Create and populate keyword tables
    # This stores the PET keywords we'll use to search patent text
    exec_sql(q("""
        -- Create temp table for storing keyword terms
        IF OBJECT_ID('tempdb..#kw','U') IS NOT NULL DROP TABLE #kw;
        CREATE TABLE #kw(term varchar(200) NOT NULL PRIMARY KEY);
    """))

    # Use fast_executemany for efficient bulk insert of keywords
    cur.fast_executemany = True
    cur.executemany("INSERT INTO #kw(term) VALUES (?)", [(t,) for t in level1_terms])
    logger.info('Inserted %d keywords into #kw temp table', len(level1_terms))

    # Create keyword LIKE patterns for fuzzy text matching
    # This handles variations in spacing and hyphens (e.g., "zero-knowledge" -> "zero%knowledge%")
    exec_sql(q("""
        -- Create temp table for LIKE patterns derived from keywords
        IF OBJECT_ID('tempdb..#kw_like','U') IS NOT NULL DROP TABLE #kw_like;
        CREATE TABLE #kw_like(term varchar(200) NOT NULL PRIMARY KEY, like_pat varchar(400) NOT NULL);

        -- Generate LIKE patterns: replace hyphens and spaces with wildcards
        -- Example: "zero-knowledge" becomes "%zero%knowledge%"
        -- This allows matching variations like "zero knowledge" or "zeroknowledge"
        INSERT INTO #kw_like(term, like_pat)
        SELECT term,
               '%' + REPLACE(REPLACE(term,'-',' '),' ','%') + '%'
        FROM #kw;
    """))

    # Search titles and abstracts for keywords
    # This identifies patents that mention PET-related terms in their text
    exec_sql(q("""
        -- Create temp table to store text-based matches (keyword approach)
        IF OBJECT_ID('tempdb..#pet_text','U') IS NOT NULL DROP TABLE #pet_text;
        CREATE TABLE #pet_text(
            appln_id bigint NOT NULL,
            match_term varchar(200) NOT NULL,
            CONSTRAINT PK_pet_text PRIMARY KEY (appln_id, match_term)
        );

        -- Search patent titles for keyword matches
        -- Uses LIKE patterns to match keywords with spacing/hyphen variations
        INSERT INTO #pet_text(appln_id, match_term)
        SELECT DISTINCT r.appln_id, k.term
        FROM #appln_year r
        JOIN SPRING25_APPLN_TITLE t ON t.appln_id = r.appln_id
        JOIN #kw_like k ON t.appln_title LIKE k.like_pat
        OPTION (RECOMPILE);  -- Recompile for better query plan with current data
    """))

    if use_abstracts:
        # Search abstracts in addition to titles (optional, slower but higher recall)
        exec_sql(q("""
            -- Search patent abstracts for keyword matches
            -- Only insert if not already found in title (avoid duplicates)
            INSERT INTO #pet_text(appln_id, match_term)
            SELECT DISTINCT r.appln_id, k.term
            FROM #appln_year r
            JOIN SPRING25_APPLN_ABSTR ab ON ab.appln_id = r.appln_id
                JOIN #kw_like k ON ab.appln_abstract LIKE k.like_pat
            WHERE NOT EXISTS (
                -- Skip if already matched in title search
                SELECT 1 FROM #pet_text pt
                WHERE pt.appln_id = r.appln_id AND pt.match_term = k.term
            )
            OPTION (RECOMPILE);  -- Recompile for better query plan with current data
        """))

    # IPC classification filtering (for all patents, especially CN/JP)
    # to identify patents in PET-relevant technology classes
    exec_sql(q("""
        -- Create temp table for IPC-based patent matches (classification approach)
        IF OBJECT_ID('tempdb..#pet_ipc','U') IS NOT NULL DROP TABLE #pet_ipc;
        SELECT DISTINCT a.appln_id
        INTO #pet_ipc
        FROM SPRING25_APPLN a
        JOIN SPRING25_APPLN_IPC i ON i.appln_id = a.appln_id
        JOIN #appln_year r ON r.appln_id = a.appln_id
        WHERE i.class_symbol LIKE 'H04L9/3%'      -- Cryptographic mechanisms (encryption)
           OR i.class_symbol LIKE 'H04L29/06%'    -- Network security protocols (legacy)
           OR i.class_symbol LIKE 'G06F21/62%'    -- Data protection/access control
           OR i.class_symbol LIKE 'G06F21/75%'    -- Privacy-preserving hardware/software
           OR i.class_symbol LIKE 'G06N3/08%';    -- Neural networks (AI/ML for PETs)

        -- Create index for efficient joins with other temp tables
        CREATE INDEX IX_pet_ipc_appln ON #pet_ipc(appln_id);
    """))

    # CPC classification filtering (for US/EU/KR patents, includes H04L63)
    # to identify patents in PET-relevant technology classes
    # H04L63 (network security) replaces older H04L29/06 in CPC
    exec_sql(q("""
        -- Create temp table for CPC-based patent matches
        IF OBJECT_ID('tempdb..#pet_cpc','U') IS NOT NULL DROP TABLE #pet_cpc;
        SELECT DISTINCT a.appln_id
        INTO #pet_cpc
        FROM SPRING25_APPLN a
        JOIN SPRING25_APPLN_CPC c ON c.appln_id = a.appln_id
        JOIN #appln_year r ON r.appln_id = a.appln_id
        WHERE REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'H04L9/3%'   -- Cryptographic mechanisms (encryption)
           OR REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'H04L63/%'   -- Network security (CPC-specific)
           OR REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'G06F21/60%' -- Data anonymization
           OR REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'G06F21/62%' -- Data protection/access control
           OR REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'G06F21/75%' -- Privacy-preserving hardware/software
           OR REPLACE(c.cpc_class_symbol, ' ', '') LIKE 'G06N3/08%'; -- Neural networks/deep learning

        -- Create index for efficient joins with other temp tables
        CREATE INDEX IX_pet_cpc_appln ON #pet_cpc(appln_id);
    """))

    # Combine IPC and CPC classifications (union for maximum coverage)
    exec_sql(q("""
        -- Merge IPC and CPC matches into single classification table
        IF OBJECT_ID('tempdb..#pet_classification','U') IS NOT NULL DROP TABLE #pet_classification;
        SELECT DISTINCT appln_id
        INTO #pet_classification
        FROM (
            -- UNION combines both IPC and CPC matches without duplicates
            SELECT appln_id FROM #pet_ipc
            UNION
            SELECT appln_id FROM #pet_cpc
        ) combined;

        -- Create index for efficient joins in final export queries
        CREATE INDEX IX_pet_class_appln ON #pet_classification(appln_id);
    """))

    def read_sql_df(sql):
        """Execute SQL and return results as pandas DataFrame.

        Args:
            sql: SQL query to execute

        Returns:
            DataFrame containing query results
        """
        return pd.read_sql(q(sql), conn)

    # Close cursor to avoid "connection is busy" errors when pandas reads
    # Pandas will create its own cursor for executing queries
    cur.close()

    # Export three complementary patent identification strategies
    # Strategy 1: Keyword-only - patents with PET terms in text (high recall)
    pet_text = read_sql_df("SELECT appln_id, match_term FROM #pet_text;")

    # Strategy 2: Intersection - patents with both keywords AND classification codes (high precision)
    # This combines text evidence with classification evidence for higher confidence
    pet_intersection = read_sql_df("""
        SELECT t.appln_id, t.match_term
        FROM #pet_text t
        JOIN #pet_classification i ON i.appln_id = t.appln_id;
    """)

    # Strategy 3: Classification-only - patents in PET classes without keyword matches (baseline)
    pet_classification_only = read_sql_df("SELECT appln_id FROM #pet_classification;")

    # Helper function to append data to consolidated files
    def append_to_parquet(df, filename):
        """Append dataframe to parquet file, or create if doesn't exist.

        This enables year-by-year processing with resumability. Each year's data
        is appended to a single consolidated file rather than creating separate files.

        Args:
            df: DataFrame to append
            filename: Name of parquet file (will be placed in out_dir)
        """
        filepath = os.path.join(out_dir, filename)
        if os.path.exists(filepath):
            # Read existing data and append new rows
            existing_df = pd.read_parquet(filepath)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(filepath)
            logger.info('Appended %d rows to %s (total: %d)', len(df), filename, len(combined_df))
        else:
            # Create new file for first year
            df.to_parquet(filepath)
            logger.info('Created %s with %d rows', filename, len(df))

    # Save identification datasets (append to single consolidated files)
    # These files track which identification strategy found each patent
    append_to_parquet(pet_text, "pet_ids_keyword_only.parquet")
    append_to_parquet(pet_intersection, "pet_ids_intersection.parquet")
    append_to_parquet(pet_classification_only, "pet_ids_classification_only.parquet")

    # Use keyword-only approach for full dataset export (broadest coverage)
    # This ensures we capture all potentially relevant patents for downstream analysis
    PET_IDS = pet_text['appln_id'].drop_duplicates()
    patent_count = len(PET_IDS)

    if patent_count == 0:
        logger.info('No patents found for year %d', year)
        return 0

    logger.info('Found %d patents for year %d', patent_count, year)

    def chunks(iterable, size):
        """Split an iterable into fixed-size chunks for batch processing.

        Args:
            iterable: Any iterable to split into chunks
            size: Maximum size of each chunk

        Yields:
            Lists of items, each containing up to 'size' elements
        """
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    def export_by_ids(filename, select_tmpl, id_series, batch=5000):
        """Export data by IDs and append to consolidated file.

        Processes IDs in batches to avoid SQL query size limits and memory issues.
        The select_tmpl should contain {IDLIST} placeholder for ID substitution.

        Args:
            filename: Name of parquet file to create/append
            select_tmpl: SQL query template with {IDLIST} placeholder
            id_series: Pandas Series of patent IDs to export
            batch: Batch size for ID processing (default: 5000)

        Returns:
            Combined DataFrame with all exported data
        """
        outs = []
        # Process IDs in batches of 5000 to avoid query length limits
        for ids in chunks(id_series.tolist(), batch):
            # Build comma-separated ID list for SQL IN clause
            id_list = ",".join(str(int(x)) for x in ids)
            df = read_sql_df(select_tmpl.format(IDLIST=id_list))
            outs.append(df)

        # Combine all batches into single dataframe
        if outs:
            out = pd.concat(outs, ignore_index=True)
        else:
            out = pd.DataFrame()

        # Append to consolidated file instead of year-specific directory
        append_to_parquet(out, filename)
        return out

    # Export all related data tables for the identified PET patents
    # Each export captures a different dimension of patent data

    # Export 1: Core application metadata
    # Contains basic patent information: IDs, filing authority, dates, and family links
    export_by_ids("appln_core.parquet", f"""
        SELECT a.appln_id,                              -- Unique patent application ID
               a.appln_auth,                            -- Filing authority (e.g., 'US', 'EP', 'JP')
               a.appln_nr,                              -- Application number
               a.appln_kind,                            -- Application kind (e.g., 'A' for patent). Not used for filtering since we focus on families
               a.appln_filing_date,                     -- Date patent was filed
               a.appln_filing_year,                     -- Year patent was filed
               a.earliest_publn_date,                   -- First publication date
               a.earliest_publn_year,                   -- First publication year
               pr.prio_year,                            -- Priority year (earliest filing)
               a.docdb_family_id,                       -- DOCDB family ID (broader grouping)
               a.inpadoc_family_id                      -- INPADOC family ID (narrower grouping)
        FROM SPRING25_APPLN a
        LEFT JOIN SPRING25_PRIO_APP pr ON pr.appln_id = a.appln_id
        WHERE a.appln_id IN ({{IDLIST}})
        ORDER BY a.appln_id
    """, PET_IDS)

    # Export 2: Patent text (titles and abstracts)
    # Contains the textual content used for keyword matching
    export_by_ids("title_abstract.parquet", """
        SELECT p.appln_id,
            t.appln_title,                              -- Patent title
            ab.appln_abstract                           -- Patent abstract (may be NULL)
        FROM (SELECT appln_id FROM SPRING25_APPLN WHERE appln_id IN ({IDLIST})) p
        LEFT JOIN SPRING25_APPLN_TITLE  t  ON t.appln_id  = p.appln_id
        LEFT JOIN SPRING25_APPLN_ABSTR  ab ON ab.appln_id = p.appln_id
        ORDER BY p.appln_id
    """, PET_IDS)

    # Export 3: IPC classification codes
    # International Patent Classification - technology categories assigned to patents
    # One patent can have multiple IPC codes
    export_by_ids("ipc.parquet", """
        SELECT i.appln_id,
               i.class_symbol AS ipc_class_symbol       -- IPC code (e.g., 'H04L9/32')
        FROM SPRING25_APPLN_IPC i
        WHERE i.appln_id IN ({IDLIST})
        ORDER BY i.appln_id, i.class_symbol
    """, PET_IDS)

    # Export 4: CPC classification codes
    # Cooperative Patent Classification - more granular than IPC
    # Used by EPO and USPTO, one patent can have multiple CPC codes
    export_by_ids("cpc.parquet", """
        SELECT c.appln_id,
               c.cpc_class_symbol                       -- CPC code (e.g., 'H04L63/04')
        FROM SPRING25_APPLN_CPC c
        WHERE c.appln_id IN ({IDLIST})
        ORDER BY c.appln_id, c.cpc_class_symbol
    """, PET_IDS)

    # Export 5: WIPO technology fields
    # WIPO 35-field classification for technology areas
    # Field_share indicates the fractional contribution of each field to the patent
    export_by_ids("wipo.parquet", """
        SELECT w.Appln_id AS appln_id,
               w.Field_code,                            -- WIPO field code (1-35)
               w.Field_share                            -- Fractional share (sum to 1.0 per patent)
        FROM SPRING25_TECH_WIPO w
        WHERE w.Appln_id IN ({IDLIST})
        ORDER BY w.Appln_id
    """, PET_IDS)

    # Export 6: Patent quality indicators
    # EPO-normalized quality metrics for patent significance assessment
    export_by_ids("quality_epo.parquet", """
        SELECT q.Appln_id AS appln_id,
               q.Fwd_Cits5,                             -- Forward citations (5-year window)
               q.Family_Size,                           -- Number of family members
               q.Claims                                 -- Number of claims
        FROM SPRING25_QUALITY_EPO_NORM q
        WHERE q.Appln_id IN ({IDLIST})
        ORDER BY q.Appln_id
    """, PET_IDS)

    # Export 7: Geographic data (inventors and applicants)
    # Links patents to inventors/applicants with country and regional codes
    # Used for geographic distribution and collaboration analysis
    export_by_ids("inventor_geo.parquet", """
        SELECT pa.appln_id,
               pr.Ctry_code AS person_ctry_code,        -- Country code (ISO 2-letter)
               pr.NUTS3_CODE AS nuts3_code,             -- NUTS3 regional code (EU regions)
               CASE
                   WHEN pa.invt_seq_nr > 0 THEN 'inventor'    -- Person is inventor
                   WHEN pa.applt_seq_nr > 0 THEN 'applicant'  -- Person is applicant
                   ELSE 'unknown'
               END AS person_role
        FROM SPRING25_PERS_APPLN pa
        JOIN SPRING25_PERSON_REGPAT pr ON pr.Person_id = pa.person_id
        WHERE pa.appln_id IN ({IDLIST})
        ORDER BY pa.appln_id
    """, PET_IDS)

    # Export 8: Company linkage (HAN-ORBIS matching)
    # Links patents to companies in the ORBIS database via harmonized names
    # Enables firm-level analysis of patent portfolios
    export_by_ids("han_orbis.parquet", """
        SELECT pa.appln_id,
               ho.bvd_id,                               -- Bureau van Dijk company ID
               ho.count_match                           -- Match confidence score
        FROM SPRING25_PERS_APPLN pa
        JOIN SPRING25_HAN_ORBIS ho ON ho.person_id = pa.person_id
        WHERE pa.appln_id IN ({IDLIST})
        ORDER BY pa.appln_id
    """, PET_IDS)

    logger.info('Completed export for year %d: %d patents', year, patent_count)
    return patent_count


def run_export_resumable(
    dsn_name: str,
    db_name: str,
    out_dir: str,
    year_from: int = 2010,
    year_to: int = 2025,
    use_abstracts: bool = True,
    keywords_csv: str = None,
    reset_state: bool = False
):
    """Run the PATSTAT export pipeline with year-by-year resumability.

    This is the main entry point for PET patent extraction from PATSTAT.
    It implements a resumable ETL process that can recover from failures
    by tracking which years have been successfully processed.

    The function:
    1. Loads PET keywords from CSV
    2. Checks which years are already completed (via state tracking)
    3. Processes each remaining year sequentially
    4. Appends results to consolidated parquet files
    5. Tracks completion state after each year

    If interrupted (error, Ctrl+C, etc.), the process can be restarted
    and will automatically skip completed years.

    Args:
        dsn_name: ODBC DSN name for PATSTAT database connection
        db_name: Database name (e.g., 'patstat_2025_1')
        out_dir: Output directory for parquet files and state tracking
        year_from: Start year for priority year filter (inclusive)
        year_to: End year for priority year filter (inclusive)
        use_abstracts: Include abstract search (slower but higher recall).
                       If False, only searches titles.
        keywords_csv: Path to keywords CSV. If None, uses default gold standard
                      from src/keywords/data/gold/level1_keywords.csv
        reset_state: If True, clear previous completion state and start fresh.
                     Use this to reprocess all years from scratch.

    Returns:
        Output directory path containing consolidated parquet files
    """
    # Ensure output directory exists
    ensure_out_dir(out_dir)

    # Initialize state tracker for resumability
    # This JSON file tracks which years have been successfully completed
    state_file = os.path.join(out_dir, '.etl_state.json')
    state = ETLState(state_file)

    if reset_state:
        # Clear all previous state and start from scratch
        logger.info('Resetting state: starting fresh')
        state.reset()

    # Load PET keywords once at the start (shared across all years)
    # These keywords define what we consider to be privacy-enhancing technology
    LEVEL1_TERMS = load_level1_keywords(keywords_csv)
    logger.info('Loaded %d level-1 terms', len(LEVEL1_TERMS))

    # Determine which years need processing based on state tracking
    # This allows resuming from the last incomplete year after failures
    remaining_years = state.get_remaining_years(year_from, year_to)

    if not remaining_years:
        # All years have been completed - nothing to do
        logger.info('All years already completed! Years: %d-%d', year_from, year_to)
        print(f"\nAll years ({year_from}-{year_to}) have already been processed!")
        print(f"Total patents extracted: {state.total_patents_extracted}")
        print("\nTo start fresh, use --reset-state flag")
        return out_dir

    # Log processing plan
    logger.info('Starting PATSTAT resumable export: DSN=%s DB=%s OUT=%s years=%s-%s abstracts=%s',
                dsn_name, db_name, out_dir, year_from, year_to, use_abstracts)
    logger.info('Years to process: %s', remaining_years)
    logger.info('Previously completed years: %s', sorted(list(state.completed_years)))

    # Display processing plan to user
    print(f"\nResumable mode: Data will be appended to single consolidated files")
    print(f"Years to process: {len(remaining_years)} ({remaining_years[0]}-{remaining_years[-1]})")
    if state.completed_years:
        print(f"Previously completed: {len(state.completed_years)} years")
        print(f"Already extracted: {state.total_patents_extracted} patents")

    def connect():
        """Create ODBC connection to PATSTAT database with autocommit enabled.

        Returns:
            Active pyodbc connection with autocommit=True
        """
        return pyodbc.connect(f"DSN={dsn_name};", autocommit=True)

    # Process each remaining year sequentially
    # Uses a persistent connection to avoid reconnection overhead
    with connect() as conn:
        for i, year in enumerate(remaining_years, 1):
            try:
                # Display progress banner
                print(f"\n{'='*60}")
                print(f"Processing year {year} ({i}/{len(remaining_years)})")
                print(f"{'='*60}")

                # Extract patents for this year
                # This calls run_export_single_year which handles all SQL logic
                patent_count = run_export_single_year(
                    conn=conn,
                    db_name=db_name,
                    year=year,
                    out_dir=out_dir,
                    level1_terms=LEVEL1_TERMS,
                    use_abstracts=use_abstracts
                )

                # Mark year as complete in state tracker
                # This ensures we can resume from the next year if interrupted
                state.mark_year_complete(year, patent_count)
                print(f"✓ Year {year} completed: {patent_count} patents extracted")

            except Exception as e:
                # Log error and provide recovery instructions
                logger.error('Failed to process year %d: %s', year, e)
                print(f"\n{'='*60}")
                print(f"✗ Error processing year {year}")
                print(f"{'='*60}")
                print(f"Error: {e}")
                print(f"\nProgress saved. You can restart the script to resume from year {year}")
                # Re-raise to stop processing (state is already saved)
                raise

    # All years completed successfully
    logger.info('Export complete. Files saved to %s', out_dir)
    summary = state.get_summary()

    # Display completion summary to user
    print(f"\n{'='*60}")
    print("✓ Export completed successfully!")
    print(f"{'='*60}")
    print(f"Years processed: {year_from}-{year_to}")
    print(f"Total patents extracted: {summary['total_patents_extracted']}")
    print(f"Output directory: {out_dir}")
    print(f"\nAll data saved in consolidated parquet files")
    print(f"{'='*60}")

    return out_dir
