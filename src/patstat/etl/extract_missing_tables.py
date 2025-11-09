"""Extract missing PATSTAT tables for enhanced PET patent analysis.

This script extracts additional data from PATSTAT SPRING25 schema to address
critical gaps identified in MISSING_DATA_ANALYSIS.md:

Priority 1: Geographic coverage (18.5% → 90-95%)
    - PERSON_INV_CTRY, PERSON_APP_CTRY, PERSON
Priority 2: Institutional classification (NEW)
    - INST
Priority 3: Firm linkage (30% → 100%)
    - HAN, HAN_NAMES
Priority 4: Citation network (NEW)
    - CITATION
Priority 5: Patent lifecycle (NEW)
    - INPADOC_LEGAL_EVENT
Priority 6: ICT classification (NEW)
    - ICT_TECH

Usage:
    python extract_missing_tables.py --dsn PATSTAT --db patstat_2025_1
    python extract_missing_tables.py --priorities 1,2,3  # Only specific priorities
    python extract_missing_tables.py --dry-run  # Preview queries only
"""

import pyodbc
import pandas as pd
import os
import argparse
import itertools
import logging
from pathlib import Path
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MissingTablesExtractor:
    """Extract enrichment tables from PATSTAT to enhance PET patent analysis.

    This class extracts 6 categories of additional data ("enrichment tables")
    that are not included in the core extraction (extract_patstat.py):

    Priority 1: Geographic data - Enhanced country coverage (90-95% vs 18.5%)
    Priority 2: Institutional classification - Organization types (universities, companies, etc.)
    Priority 3: Firm linkage - Harmonized applicant names for company analysis
    Priority 4: Citation network - Forward and backward citation relationships
    Priority 5: Legal events - Patent lifecycle events (grants, renewals, abandonments)
    Priority 6: ICT classification - ICT technology categorization

    The extractor operates on patent IDs from appln_core.parquet and uses
    batch processing to handle large datasets efficiently.

    Typical workflow:
        extractor = MissingTablesExtractor(dsn='PATSTAT', db_name='patstat_2025_1',
                                          out_dir='data/raw')
        extractor.run(source_file='data/raw/appln_core.parquet', priorities=[1,2,3])
    """

    def __init__(self, dsn: str, db_name: str, out_dir: str, batch_size: int = 5000):
        """Initialize the enrichment tables extractor.

        Args:
            dsn: ODBC data source name configured in system
            db_name: Database name in PATSTAT (e.g., 'patstat_2025_1')
            out_dir: Output directory where parquet files will be saved
            batch_size: Number of patent IDs to process per SQL query
                       (default 5000, tune based on database performance)

        Note:
            The output directory is created automatically if it doesn't exist.
        """
        self.dsn = dsn
        self.db_name = db_name
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.conn = None

        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def connect(self):
        """Establish database connection."""
        conn_str = f'DSN={self.dsn};Database={self.db_name}'
        self.conn = pyodbc.connect(conn_str, autocommit=True)
        logger.info(f"Connected to {self.dsn}/{self.db_name}")

    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")

    def load_patent_ids(self, source_file: str) -> pd.Series:
        """Load patent IDs from appln_core.parquet.

        Args:
            source_file: Path to appln_core.parquet

        Returns:
            Series of unique patent application IDs
        """
        logger.info(f"Loading patent IDs from {source_file}")
        df = pd.read_parquet(source_file)
        appln_ids = df['appln_id'].drop_duplicates().sort_values()
        logger.info(f"Loaded {len(appln_ids):,} unique patent IDs")
        return appln_ids

    def chunks(self, iterable, size):
        """Split an iterable into fixed-size chunks for batch processing.

        Yields consecutive chunks from the iterable until exhausted.
        Useful for processing large ID lists in manageable batches.

        Args:
            iterable: Any iterable (list, Series, etc.)
            size: Maximum number of items per chunk

        Yields:
            Lists containing up to 'size' items from the iterable
        """
        it = iter(iterable)
        while True:
            batch = list(itertools.islice(it, size))
            if not batch:
                break
            yield batch

    def execute_batch_query(self, query_template: str, appln_ids: pd.Series) -> pd.DataFrame:
        """Execute SQL query in batches and combine results into single DataFrame.

        Splits patent IDs into batches to avoid SQL query length limits,
        executes the query for each batch, and combines results.

        Args:
            query_template: SQL query with {IDLIST} placeholder for ID substitution
                           Example: "SELECT * FROM table WHERE id IN ({IDLIST})"
            appln_ids: Pandas Series of patent IDs to query

        Returns:
            Combined DataFrame containing all batch results, or empty DataFrame
            if no data was retrieved

        Note:
            Progress is logged for each batch (batch X/Y, rows retrieved).
            Batch size is controlled by self.batch_size (default 5000).
        """
        results = []
        total_batches = (len(appln_ids) + self.batch_size - 1) // self.batch_size

        for i, id_batch in enumerate(self.chunks(appln_ids.tolist(), self.batch_size), 1):
            id_list = ",".join(str(int(x)) for x in id_batch)
            query = query_template.format(IDLIST=id_list)

            logger.info(f"Executing batch {i}/{total_batches} ({len(id_batch)} IDs)")

            cursor = self.conn.cursor()
            cursor.execute(query)

            # Fetch results before closing cursor
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            cursor.close()

            # Convert to DataFrame
            if rows:
                batch_df = pd.DataFrame.from_records(rows, columns=columns)
                results.append(batch_df)
                logger.info(f"  Retrieved {len(batch_df):,} rows")

        if results:
            combined = pd.concat(results, ignore_index=True)
            logger.info(f"Total rows retrieved: {len(combined):,}")
            return combined
        else:
            logger.warning("No data retrieved")
            return pd.DataFrame()

    def extract_priority_1_geography(self, appln_ids: pd.Series):
        """Extract Priority 1: Geographic data.

        Extracts from:
        - PERSON_INV_CTRY (inventor countries)
        - PERSON_APP_CTRY (applicant countries)
        - PERSON (raw person data with addresses)

        Output: person_geography_complete.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 1: GEOGRAPHIC DATA EXTRACTION")
        logger.info("=" * 80)

        query = """
        -- Comprehensive geographic data extraction for inventors and applicants
        -- Combines three sources of country data to maximize coverage
        SELECT DISTINCT
            pa.appln_id,                             -- Patent application ID
            pa.person_id,                            -- Unique person identifier
            -- Determine role based on sequence numbers
            CASE
                WHEN pa.invt_seq_nr > 0 THEN 'inventor'    -- Person is an inventor
                WHEN pa.applt_seq_nr > 0 THEN 'applicant'  -- Person is an applicant
                ELSE 'unknown'                              -- Neither (rare edge case)
            END AS person_role,
            ic.ctry_code AS inventor_ctry_code,      -- Country code from inventor-specific table (REGPAT quality)
            ac.Ctry_Code AS applicant_ctry_code,     -- Country code from applicant-specific table (REGPAT quality)
            p.person_ctry_code AS person_ctry_code,  -- Country code from raw PERSON table (fallback)
            p.person_name,                            -- Raw person/organization name
            p.person_address                          -- Raw address (may contain country info)
        FROM SPRING25_PERS_APPLN pa
        -- LEFT JOIN to inventor countries (REGPAT-enhanced geocoding)
        LEFT JOIN SPRING25_PERSON_INV_CTRY ic
            ON ic.Appln_id = pa.appln_id AND ic.person_id = pa.person_id
        -- LEFT JOIN to applicant countries (REGPAT-enhanced geocoding)
        LEFT JOIN SPRING25_PERSON_APP_CTRY ac
            ON ac.Appln_id = pa.appln_id AND ac.Person_id = pa.person_id
        -- LEFT JOIN to raw person data (fallback for missing REGPAT data)
        LEFT JOIN SPRING25_PERSON p
            ON p.person_id = pa.person_id
        WHERE pa.appln_id IN ({IDLIST})
        ORDER BY pa.appln_id, pa.person_id
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            # Add computed field: best available country code
            df['ctry_code_best'] = df['inventor_ctry_code'].fillna(
                df['applicant_ctry_code']
            ).fillna(
                df['person_ctry_code']
            )

            output_file = self.out_dir / 'person_geography_complete.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")
            logger.info(f"  Coverage: {df['ctry_code_best'].notna().sum():,} / {len(df):,} records have country codes")

            # Summary statistics
            coverage_by_role = df.groupby('person_role')['ctry_code_best'].apply(
                lambda x: f"{x.notna().sum():,} / {len(x):,} ({x.notna().sum()/len(x)*100:.1f}%)"
            )
            logger.info(f"  Coverage by role:\n{coverage_by_role}")

        return df

    def extract_priority_2_institutions(self, appln_ids: pd.Series):
        """Extract Priority 2: Institutional classification.

        Extracts from:
        - INST (sector classification: universities, companies, government, etc.)

        Output: person_institutions.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 2: INSTITUTIONAL CLASSIFICATION")
        logger.info("=" * 80)

        query = """
        -- Extract institutional sector classification for applicants
        -- Uses OECD-REGPAT methodology to classify organizations by type
        SELECT DISTINCT
            pa.appln_id,                             -- Patent application ID
            inst.person_id,                          -- Unique person/organization identifier
            -- Sector classification flags (values 0-1, can sum > 1 for mixed types)
            inst.Individual,                         -- Individual inventor/applicant
            inst.Companies,                          -- Private companies/corporations
            inst.Government,                         -- Government organizations
            inst.Universities,                       -- Universities and research institutions
            inst.Hospitals,                          -- Hospitals and medical centers
            inst.PNP,                                -- Private non-profit organizations
            inst.Unknown                             -- Unclassified organizations
        FROM SPRING25_PERS_APPLN pa
        -- INNER JOIN: Only include applicants with institutional classification
        JOIN SPRING25_INST inst ON inst.person_id = pa.person_id
        WHERE pa.appln_id IN ({IDLIST})
          AND pa.applt_seq_nr > 0  -- Only applicants (not inventors)
        ORDER BY pa.appln_id, inst.person_id
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            output_file = self.out_dir / 'person_institutions.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")

            # Summary statistics
            for col in ['Individual', 'Companies', 'Government', 'Universities', 'Hospitals', 'PNP', 'Unknown']:
                count = df[col].sum() if col in df.columns else 0
                logger.info(f"  {col}: {count:,}")

        return df

    def extract_priority_3_firm_linkage(self, appln_ids: pd.Series):
        """Extract Priority 3: Harmonized applicant names.

        Extracts from:
        - HAN (harmonized names)
        - HAN_NAMES (harmonized names with geography)

        Output: han_applicants_complete.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 3: FIRM LINKAGE (HARMONIZED NAMES)")
        logger.info("=" * 80)

        query = """
        -- Extract harmonized applicant names (HAN) for firm-level analysis
        -- Links raw person names to standardized organization identifiers
        SELECT DISTINCT
            pa.appln_id,                             -- Patent application ID
            h.Person_id AS person_id,                -- Raw person ID
            h.HAN_ID AS han_id,                      -- Harmonized applicant name ID (groups variants)
            h.Person_name_clean AS han_name,         -- Cleaned version of raw person name
            hn.HAN_Person_name AS han_harmonized_name, -- Standardized canonical name for this HAN_ID
            -- Use country from HAN_NAMES table if available, else from HAN table
            COALESCE(hn.Person_ctry_Code, h.Person_ctry_code) AS han_ctry_code
        FROM SPRING25_PERS_APPLN pa
        -- INNER JOIN: Only include applicants with harmonized names
        JOIN SPRING25_HAN h ON h.Person_id = pa.person_id
        -- LEFT JOIN to get canonical harmonized name and geography
        -- Not all HAN_IDs have entries in HAN_NAMES (some have only HAN)
        LEFT JOIN SPRING25_HAN_NAMES hn ON hn.HAN_ID = h.HAN_ID
        WHERE pa.appln_id IN ({IDLIST})
          AND pa.applt_seq_nr > 0  -- Only applicants (not inventors)
        ORDER BY pa.appln_id, h.Person_id
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            output_file = self.out_dir / 'han_applicants_complete.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")
            logger.info(f"  Unique organizations (HAN_ID): {df['han_id'].nunique():,}")
            logger.info(f"  With country codes: {df['han_ctry_code'].notna().sum():,}")

        return df

    def extract_priority_4_citations(self, appln_ids: pd.Series):
        """Extract Priority 4: Citation network.

        Extracts from:
        - CITATION (patent citations - both forward and backward)

        Output: citations_complete.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 4: CITATION NETWORK")
        logger.info("=" * 80)

        query = """
        -- Extract both forward and backward citations for network analysis
        -- Forward citations: Our PET patents cite other patents
        -- Backward citations: Other patents cite our PET patents

        -- PART 1: Forward citations (our patents as citing documents)
        SELECT
            pp.appln_id AS citing_appln_id,          -- Our PET patent doing the citing
            c.cited_appln_id,                        -- Patent being cited (may be NULL for NPL)
            c.citn_origin,                           -- Origin: 'ISR', 'SEARCH', 'APPLICANT', 'EXAMINER', etc.
            c.citn_id,                               -- Unique citation ID
            c.pat_citn_seq_nr,                       -- Sequence number for patent citations
            c.npl_publn_id,                          -- Non-patent literature ID (scientific papers, etc.)
            c.npl_citn_seq_nr,                       -- Sequence number for NPL citations
            'forward' AS citation_direction          -- Direction label for analysis
        FROM SPRING25_PAT_PUBLN pp
        -- JOIN to get citations made BY our patents
        JOIN SPRING25_CITATION c ON c.pat_publn_id = pp.pat_publn_id
        WHERE pp.appln_id IN ({IDLIST})

        UNION ALL

        -- PART 2: Backward citations (other patents citing our patents)
        SELECT
            pp.appln_id AS citing_appln_id,          -- Our PET patent being cited
            c.cited_appln_id,                        -- Should match pp.appln_id (our patent)
            c.citn_origin,                           -- Origin of citation
            c.citn_id,                               -- Unique citation ID
            c.pat_citn_seq_nr,                       -- Sequence number
            c.npl_publn_id,                          -- NPL reference (usually NULL for backward)
            c.npl_citn_seq_nr,                       -- NPL sequence (usually NULL for backward)
            'backward' AS citation_direction         -- Direction label for analysis
        FROM SPRING25_CITATION c
        -- JOIN where cited patent is one of our PET patents
        JOIN SPRING25_PAT_PUBLN pp ON pp.pat_publn_id = c.cited_pat_publn_id
        WHERE pp.appln_id IN ({IDLIST})

        ORDER BY citing_appln_id, cited_appln_id
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            output_file = self.out_dir / 'citations_complete.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")
            logger.info(f"  Forward citations: {(df['citation_direction'] == 'forward').sum():,}")
            logger.info(f"  Backward citations: {(df['citation_direction'] == 'backward').sum():,}")
            logger.info(f"  With NPL references: {df['npl_publn_id'].notna().sum():,}")

        return df

    def extract_priority_5_legal_events(self, appln_ids: pd.Series):
        """Extract Priority 5: Patent lifecycle events.

        Extracts from:
        - INPADOC_LEGAL_EVENT (grants, abandonments, renewals, etc.)

        Output: legal_events.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 5: LEGAL EVENTS (PATENT LIFECYCLE)")
        logger.info("=" * 80)

        query = """
        -- Extract patent lifecycle events (grants, renewals, abandonments, etc.)
        -- Uses INPADOC legal event data for comprehensive lifecycle tracking
        SELECT
            le.appln_id,                             -- Patent application ID
            le.event_seq_nr,                         -- Sequence number (multiple events per patent)
            le.event_auth,                           -- Authority code (e.g., 'EP', 'US', 'JP')
            le.event_code,                           -- Event code (authority-specific)
            le.event_filing_date,                    -- Date event was filed
            le.event_publn_date,                     -- Date event was published
            le.event_effective_date,                 -- Effective date of the event
            -- JOIN to decode event codes into human-readable descriptions
            lec.event_descr,                         -- Description of event (e.g., 'Patent granted')
            lec.event_category_code,                 -- Category code (e.g., 'GR' for grant)
            lec.event_category_title                 -- Category title (e.g., 'Grant')
        FROM SPRING25_INPADOC_LEGAL_EVENT le
        -- LEFT JOIN: Not all event codes have standardized descriptions
        LEFT JOIN SPRING25_LEGAL_EVENT_CODE lec
            ON lec.event_auth = le.event_auth
            AND lec.event_code = le.event_code
        WHERE le.appln_id IN ({IDLIST})
        -- Order chronologically by filing date
        ORDER BY le.appln_id, le.event_filing_date
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            output_file = self.out_dir / 'legal_events.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")

            # Summary by event category
            if 'event_category_title' in df.columns:
                top_categories = df['event_category_title'].value_counts().head(10)
                logger.info(f"  Top event categories:\n{top_categories}")

        return df

    def extract_priority_6_ict_classification(self, appln_ids: pd.Series):
        """Extract Priority 6: ICT technology classification.

        Extracts from:
        - ICT_TECH (ICT classification codes and shares)

        Output: ict_classification.parquet
        """
        logger.info("=" * 80)
        logger.info("PRIORITY 6: ICT TECHNOLOGY CLASSIFICATION")
        logger.info("=" * 80)

        query = """
        -- Extract ICT technology classification codes
        -- ICT classification identifies information and communication technology patents
        -- Uses fractional counting when a patent spans multiple ICT categories
        SELECT
            ict.Appln_id AS appln_id,                -- Patent application ID
            ict.ICT_Code AS ict_code,                -- Main ICT category code (e.g., '1', '2', '3')
            ict.ICT_Sub_Code AS ict_sub_code,        -- ICT subcategory code for finer granularity
            ict.ICT_Main AS ict_main,                -- Flag: 1 if this is the main ICT category for this patent
            ict.ICT_Share AS ict_share,              -- Fractional share (sum to 1.0 per patent across all ICT codes)
            ict.ICT_content AS ict_content           -- Content indicator (level of ICT involvement)
        FROM SPRING25_ICT_TECH ict
        WHERE ict.Appln_id IN ({IDLIST})
        -- Order by share descending to show most relevant ICT codes first
        ORDER BY ict.Appln_id, ict.ICT_Share DESC
        """

        df = self.execute_batch_query(query, appln_ids)

        if not df.empty:
            output_file = self.out_dir / 'ict_classification.parquet'
            df.to_parquet(output_file, index=False)
            logger.info(f"✓ Saved to {output_file}")
            logger.info(f"  Unique ICT codes: {df['ict_code'].nunique()}")
            logger.info(f"  Unique ICT sub-codes: {df['ict_sub_code'].nunique()}")

            # Summary by main ICT code
            if 'ict_code' in df.columns:
                top_codes = df['ict_code'].value_counts().head(10)
                logger.info(f"  Top ICT codes:\n{top_codes}")

        return df

    def run(self, source_file: str, priorities: Optional[List[int]] = None, dry_run: bool = False):
        """Run enrichment extraction for specified priority categories.

        This is the main public method to extract enrichment tables. It:
        1. Loads patent IDs from appln_core.parquet
        2. Connects to PATSTAT database
        3. Executes extraction for each selected priority
        4. Saves results to parquet files in out_dir
        5. Disconnects from database

        Args:
            source_file: Path to appln_core.parquet containing patent IDs
                        (generated by extract_patstat.py)
            priorities: List of priority numbers to extract (1-6), or None for all.
                       Priorities:
                       1 = Geographic data (person_geography_complete.parquet)
                       2 = Institutional classification (person_institutions.parquet)
                       3 = Firm linkage (han_applicants_complete.parquet)
                       4 = Citation network (citations_complete.parquet)
                       5 = Legal events (legal_events.parquet)
                       6 = ICT classification (ict_classification.parquet)
            dry_run: If True, load IDs and print summary but don't extract data
                    (useful for estimating extraction scope)

        Raises:
            FileNotFoundError: If source_file doesn't exist
            pyodbc.Error: If database connection or queries fail

        Side Effects:
            - Creates 1-6 parquet files in self.out_dir (depending on priorities)
            - Writes progress to logger
        """
        if priorities is None:
            priorities = [1, 2, 3, 4, 5, 6]

        logger.info("=" * 80)
        logger.info("PATSTAT MISSING TABLES EXTRACTION")
        logger.info("=" * 80)
        logger.info(f"Source: {source_file}")
        logger.info(f"Output: {self.out_dir}")
        logger.info(f"Priorities: {priorities}")
        logger.info(f"Dry run: {dry_run}")
        logger.info("=" * 80)

        # Load patent IDs
        appln_ids = self.load_patent_ids(source_file)

        if dry_run:
            logger.info("DRY RUN MODE - No data will be extracted")
            logger.info(f"Would extract data for {len(appln_ids):,} patent IDs")
            return

        # Connect to database
        self.connect()

        try:
            # Execute extractions based on priorities
            if 1 in priorities:
                self.extract_priority_1_geography(appln_ids)

            if 2 in priorities:
                self.extract_priority_2_institutions(appln_ids)

            if 3 in priorities:
                self.extract_priority_3_firm_linkage(appln_ids)

            if 4 in priorities:
                self.extract_priority_4_citations(appln_ids)

            if 5 in priorities:
                self.extract_priority_5_legal_events(appln_ids)

            if 6 in priorities:
                self.extract_priority_6_ict_classification(appln_ids)

            logger.info("=" * 80)
            logger.info("✓ EXTRACTION COMPLETE")
            logger.info("=" * 80)

        finally:
            self.disconnect()


def main():
    """Command-line entry point for enrichment table extraction.

    Parses command-line arguments and orchestrates the extraction of
    enrichment tables from PATSTAT. This function is called when the
    script is run directly from the command line.

    The function:
    1. Parses CLI arguments (DSN, database, priorities, etc.)
    2. Creates MissingTablesExtractor instance
    3. Runs extraction with specified configuration
    4. Handles errors and provides user feedback

    Exit Codes:
        0 - Success
        1 - Error (invalid args, connection failure, etc.)

    Examples:
        # Extract all priorities
        python extract_missing_tables.py --dsn PATSTAT --db patstat_2025_1

        # Extract only geographic and institutional data
        python extract_missing_tables.py --priorities 1,2

        # Preview what would be extracted
        python extract_missing_tables.py --dry-run
    """
    parser = argparse.ArgumentParser(
        description='Extract missing PATSTAT tables for PET patent analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Priorities:
  1 - Geographic data (PERSON_INV_CTRY, PERSON_APP_CTRY, PERSON)
      Output: person_geography_complete.parquet (90-95% country coverage)
  2 - Institutional classification (INST)
      Output: person_institutions.parquet (universities, companies, government)
  3 - Firm linkage (HAN, HAN_NAMES)
      Output: han_applicants_complete.parquet (harmonized company names)
  4 - Citation network (CITATION)
      Output: citations_complete.parquet (forward & backward citations)
  5 - Legal events (INPADOC_LEGAL_EVENT)
      Output: legal_events.parquet (grants, renewals, abandonments)
  6 - ICT classification (ICT_TECH)
      Output: ict_classification.parquet (ICT technology categorization)

Examples:
  # Extract all priorities
  python extract_missing_tables.py --dsn PATSTAT --db patstat_2025_1

  # Extract only geographic and institutional data
  python extract_missing_tables.py --priorities 1,2

  # Preview what would be extracted
  python extract_missing_tables.py --dry-run
        """
    )

    parser.add_argument('--dsn', default='PATSTAT',
                        help='ODBC data source name (default: PATSTAT)')
    parser.add_argument('--db', default='patstat_2025_1',
                        help='Database name (default: patstat_2025_1)')
    parser.add_argument('--source',
                        default='src/patstat/data/raw/appln_core.parquet',
                        help='Source file with patent IDs')
    parser.add_argument('--out-dir',
                        default='src/patstat/data/raw',
                        help='Output directory for parquet files')
    parser.add_argument('--priorities',
                        help='Comma-separated list of priorities (1-5), e.g., "1,2,3"')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Batch size for queries (default: 5000)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be extracted without running')

    args = parser.parse_args()

    # Parse priorities
    priorities = None
    if args.priorities:
        priorities = [int(p.strip()) for p in args.priorities.split(',')]
        # Validate priorities
        if not all(1 <= p <= 6 for p in priorities):
            parser.error("Priorities must be between 1 and 6")

    # Create extractor and run
    extractor = MissingTablesExtractor(
        dsn=args.dsn,
        db_name=args.db,
        out_dir=args.out_dir,
        batch_size=args.batch_size
    )

    extractor.run(
        source_file=args.source,
        priorities=priorities,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
