"""
Transform and filter PATSTAT patent data to create PET definitions.

This script processes keyword-matched patents from extract_patstat.py and creates
two analysis-ready datasets with different precision/recall tradeoffs:

PIPELINE:
---------
Input:  data/raw/pet_ids_keyword_only.parquet (from extract_patstat.py)
        data/raw/title_abstract.parquet
        data/raw/cpc.parquet
        data/raw/ipc.parquet

Output: data/gold/pet_ids_narrow.parquet  (HIGH PRECISION - for precise statistics)
        data/gold/pet_ids_broad.parquet   (HIGH RECALL - for landscape mapping)

DEFINITIONS:
-----------
NARROW = Keywords (exact phrase, filtered) + RELEVANT_CPC_CODES
  - High precision, low false positives
  - Use for: Precise statistics

BROAD = NARROW + RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD
  - High recall, comprehensive coverage
  - Use for: Technology landscape, comprehensive analysis, emerging trends

EXACT PHRASE MATCHING:
---------------------
The SQL extraction uses wildcard matching (%word1%word2%) which causes false
positives (e.g., "on THE measuring device" matching "on device").

This script enforces EXACT phrase matching:
- Multi-word terms (2+): "on device" or "on-device" ONLY (no words between)
- Single-word terms: Word boundary matching (\bword\b)

Expected impact: Removes ~90-95% false positives

KEYWORD TIERS:
-------------
Loaded from: src/keywords/data/gold/level1_keywords_expanded.csv

- TIER_1_STRONG: High precision keywords (auto-accept)
- TIER_2_MEDIUM: Privacy-relevant technologies (all accepted, track motivation)
- TIER_3_WEAK: High false positives (require privacy OR data/ML context validation)

FILTERING STRATEGY:
------------------
1. Exact phrase match → Reject if keyword not found as exact phrase in title
2. Strong keywords (TIER_1) → Auto-accept
3. Medium keywords (TIER_2) → Accept all, track privacy vs efficiency motivation
4. Weak keywords (TIER_3) → Require privacy OR data/ML context in title

CLASSIFICATION CODES:
--------------------
Classification codes are used to ADD patents beyond keyword matches:
- RELEVANT_CPC_CODES: High-precision codes added to create NARROW definition
- RELEVANT_IPC_CODES + CPC_CODES_BROAD: Added to NARROW to create BROAD definition

For code definitions and validation, see: docs/data/CLASSIFICATION_CODES_REFERENCE.md
"""

import pandas as pd
import re
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'raw'
INTERIM_DIR = Path(__file__).parent.parent / 'data' / 'interim'
GOLD_DIR = Path(__file__).parent.parent / 'data' / 'gold'
INTERIM_DIR.mkdir(exist_ok=True, parents=True)
GOLD_DIR.mkdir(exist_ok=True, parents=True)

# Keywords source
REPO_ROOT = Path(__file__).parent.parent.parent.parent
KEYWORDS_CSV = REPO_ROOT / 'src/keywords/data/gold/level1_keywords_expanded.csv'

# Relevant PET classification codes (for weak keyword validation)
RELEVANT_IPC_CODES = [
    'G06F21/60',    # Privacy computing                                         
    'G06F21/62',    # Data protection                                      
    'G06F21/71',    # to assure secure computing or processing of information   
    'G06F21/57',    # certifying or maintaining trusted computer platforms,
    'G06F21/53',    # by executing in a restricted environment, e.g. sandbox or secure virtual machine    
    'H04W12/02',    # Privacy protection in wireless communication              
    'H04L9/00',     # Arrangements for secret or secure communication
    'H04L9/32',     # Cryptographic mechanisms or cryptographic arrangements for secret or secure communication

]

RELEVANT_CPC_CODES = [
    'G06F21/6245',  # Protecting personal data, e.g. for financial or medical purposes
    'G06F21/6254',  # by anonymising data, e.g. decorrelating personal data from the owner's identification
    #'G06F21/6272',  # registering files with third party NOTE: excluded as not core PET
    'G06N 3/098',   # Distributed learning, e.g. federated learning
    'H04L9/008',    # involving homomorphic encryption
    'H04L9/085',    # Secret sharing or secret splitting, e.g. threshold schemes
]

RELEVANT_CPC_CODES_BROAD = [
    'H04L63/04',    # for providing a confidential data exchange among entities communicating through data packet networks
    'G06F16/256',   # in federated or virtual databases
]


def load_tier_keywords(csv_path=None):
    """
    Load keyword tier assignments from level1_keywords_expanded.csv.

    Args:
        csv_path: Path to expanded keywords CSV. If None, uses default location.

    Returns:
        tuple: (strong_keywords, medium_keywords, weak_keywords, keyword_df)
               as normalized sets + original dataframe
    """
    if csv_path is None:
        csv_path = KEYWORDS_CSV

    # Load CSV
    df = pd.read_csv(csv_path)

    # Normalize keywords (uppercase, replace hyphens with spaces)
    df['normalized'] = df['keyword_term'].str.replace('-', ' ').str.upper()

    # Count words for exact matching logic
    df['word_count'] = df['keyword_term'].str.split().str.len()

    # Group by tier
    strong = set(df[df['patstat_tier'] == 'TIER_1_STRONG']['normalized'])
    medium = set(df[df['patstat_tier'] == 'TIER_2_MEDIUM']['normalized'])
    weak = set(df[df['patstat_tier'] == 'TIER_3_WEAK']['normalized'])

    return strong, medium, weak, df


class PETKeywordFilter:
    """Filter PET patents based on keyword quality and context."""

    # Tier keywords loaded from level1_keywords_expanded.csv
    # These are initialized in __init__ by calling load_tier_keywords()
    STRONG_KEYWORDS = None  # TIER_1_STRONG
    MEDIUM_KEYWORDS = None  # TIER_2_MEDIUM
    WEAK_KEYWORDS = None    # TIER_3_WEAK

    # Privacy context indicators (for medium/weak keywords)
    # CRITICAL: Require PRIVACY-SPECIFIC terminology, not generic "security" or "sensitive"
    # Remove patterns that match physical security, temperature/pressure sensors, etc.
    PRIVACY_CONTEXT = [
        # Core privacy terms (specific to data privacy, not physical security)
        r'\b(privacy[- ]preserving|privacy[- ]enhancing|privacy[- ]aware|privacy[- ]protecting)\b',
        r'\b(data\s+privacy|user\s+privacy|patient\s+privacy|consumer\s+privacy)\b',
        r'\b(privacy\s+(protection|policy|model|preserving|enhancing))\b',
        r'\bprivacy\b',  # Standalone "privacy" is OK (specific enough)

        # Anonymization and de-identification (privacy-specific)
        r'\b(anonymi[zs]ation|anonymi[zs]ed|anonymi[zs]ing)\b',
        r'\b(pseudonymi[zs]ation|pseudonymi[zs]ed|pseudonym)\b',
        r'\b(de[- ]identif|deidentif)\b',
        r'\b(k[- ]anonymity|l[- ]diversity|t[- ]closeness)\b',

        # PET-specific cryptographic techniques (not generic encryption)
        r'\b(differential\s+privacy|homomorphic\s+encryption)\b',
        r'\b(secure\s+multi[- ]?party|secret\s+sharing|zero[- ]knowledge)\b',
        r'\b(trusted\s+execution\s+environment|secure\s+enclave|TEE)\b',
        r'\b(federated\s+learning|federated\s+analytics)\b',  # Privacy-preserving federated

        # Personal/sensitive data in privacy context (require compound terms)
        r'\b(personal\s+data|personally\s+identifiable|PII)\b',
        r'\b(sensitive\s+(data|information|personal))\b',  # "sensitive data" OK, not just "sensitive"
        r'\b(patient\s+(data|privacy|information))\b',
        r'\b(user\s+(privacy|data\s+protection))\b',

        # Data protection and privacy regulations
        r'\b(data\s+protection|GDPR|privacy\s+regulation)\b',
        r'\b(privacy[- ]by[- ]design|privacy[- ]by[- ]default)\b',

        # Confidential computing (privacy-specific, not general confidentiality)
        r'\b(confidential\s+computing|encrypted\s+computation)\b',
        r'\b(private\s+(computation|learning|inference))\b',
    ]

    # Data/ML context (for medium/weak keywords)
    # CRITICAL: Require EXPLICIT ML/AI terminology to avoid matching generic engineering
    # This now requires actual machine learning/AI terms, not just "data processing"
    DATA_ML_CONTEXT = [
        # Core ML/AI terms (MANDATORY - these are what make it ML/AI)
        r'\b(machine\s+learning|deep\s+learning|neural\s+network|convolutional\s+neural)\b',
        r'\b(artificial\s+intelligence|AI\s+model|AI\s+system|AI\s+algorithm)\b',
        r'\b(model\s+(training|inference|prediction|deployment|optimization|evaluation))\b',
        r'\b((training|test|validation)\s+(data|dataset|set))\b',  # "training data" not just "training"
        r'\b(supervised\s+learning|unsupervised\s+learning|reinforcement\s+learning)\b',
        r'\b(transfer\s+learning|meta\s+learning|continual\s+learning)\b',

        # ML techniques and architectures
        # Note: Use specific "transformer" patterns to avoid electrical transformers
        r'\b(transformer\s+(model|architecture|network|layer|attention)|attention\s+mechanism)\b',
        r'\b(BERT|GPT|T5|RoBERTa|DistilBERT)\b',  # Specific transformer models
        r'\b(ResNet|VGG|AlexNet|Inception)\b',  # Vision models
        r'\b(CNN|RNN|LSTM|GRU|GAN)\b',
        r'\b(gradient\s+descent|backpropagation|optimization\s+algorithm)\b',
        r'\b(hyperparameter|overfitting|regularization|dropout)\b',

        # Federated/distributed learning (ML-specific)
        r'\b(federated\s+(learning|training|analytics|optimization))\b',
        r'\b(distributed\s+(learning|training))\b',  # NOT just "distributed computing"
        r'\b(edge\s+(learning|AI|inference))\b',  # Edge AI, NOT just "edge computing"

        # Big data analytics (not just "data processing")
        r'\b(big\s+data|data\s+(analytics|mining|science))\b',
        r'\b(predictive\s+analytics|business\s+intelligence)\b',
        r'\b(statistical\s+(learning|modeling|analysis))\b',

        # Computer vision (ML-based, not generic image processing)
        r'\b(computer\s+vision|image\s+recognition|object\s+detection)\b',
        r'\b(facial\s+recognition|image\s+classification|semantic\s+segmentation)\b',

        # NLP/speech (ML-based, not generic text/audio processing)
        r'\b(natural\s+language\s+processing|NLP|language\s+model)\b',
        r'\b(speech\s+recognition|voice\s+assistant|text\s+generation)\b',

        # ML infrastructure
        r'\b(ML\s+pipeline|model\s+serving|MLOps)\b',
        r'\b(feature\s+engineering|data\s+augmentation)\b',
    ]

    def __init__(self, csv_path=None):
        """
        Initialize filter.

        Args:
            csv_path: Path to keywords CSV (uses default if None)
        """
        # Load tier keywords from CSV (single source of truth)
        self.STRONG_KEYWORDS, self.MEDIUM_KEYWORDS, self.WEAK_KEYWORDS, self.keyword_df = load_tier_keywords(csv_path)

        print(f"Loaded tier keywords from CSV:")
        print(f"  TIER_1_STRONG: {len(self.STRONG_KEYWORDS)} keywords")
        print(f"  TIER_2_MEDIUM: {len(self.MEDIUM_KEYWORDS)} keywords")
        print(f"  TIER_3_WEAK: {len(self.WEAK_KEYWORDS)} keywords")

        # Build lookup for word counts (for exact matching)
        self.keyword_word_counts = dict(zip(
            self.keyword_df['normalized'],
            self.keyword_df['word_count']
        ))

        # OPTIMIZATION: Pre-compile all regex patterns
        print("Pre-compiling regex patterns for faster matching...")
        self.compiled_privacy_patterns = [re.compile(p, re.IGNORECASE) for p in self.PRIVACY_CONTEXT]
        self.compiled_data_ml_patterns = [re.compile(p, re.IGNORECASE) for p in self.DATA_ML_CONTEXT]

        # Build regex cache for exact phrase matching per keyword
        self.exact_phrase_regex_cache = {}
        for keyword in set(self.keyword_df['normalized']):
            word_count = self.keyword_word_counts.get(keyword, 1)
            if word_count == 1:
                self.exact_phrase_regex_cache[keyword] = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            else:
                words = keyword.split()
                pattern_parts = []
                for i, word in enumerate(words):
                    pattern_parts.append(rf'\b{re.escape(word)}\b')
                    if i < len(words) - 1:
                        pattern_parts.append(r'[\s-]')
                self.exact_phrase_regex_cache[keyword] = re.compile(''.join(pattern_parts), re.IGNORECASE)

        print(f"  Compiled {len(self.compiled_privacy_patterns)} privacy patterns")
        print(f"  Compiled {len(self.compiled_data_ml_patterns)} data/ML patterns")
        print(f"  Cached {len(self.exact_phrase_regex_cache)} exact phrase patterns")

        # Statistics
        self.strong_count = 0
        self.medium_accepted_privacy = 0  # Has privacy/data context
        self.medium_accepted_efficiency = 0  # No privacy/data context (efficiency-motivated)
        self.weak_accepted_context = 0
        self.weak_rejected = 0
        self.exact_match_rejected = 0  # New: Rejected due to exact phrase matching

    def has_privacy_context(self, text):
        """Check if text has privacy-related context (OPTIMIZED - uses pre-compiled patterns)."""
        if pd.isna(text):
            return False
        text_str = str(text)
        return any(p.search(text_str) for p in self.compiled_privacy_patterns)

    def has_data_ml_context(self, text):
        """Check if text has data/ML context (OPTIMIZED - uses pre-compiled patterns)."""
        if pd.isna(text):
            return False
        text_str = str(text)
        return any(p.search(text_str) for p in self.compiled_data_ml_patterns)

    def exact_phrase_match(self, text, keyword):
        """
        Check if keyword appears as exact phrase in text (OPTIMIZED - uses cached compiled patterns).

        For multi-word terms (2+ words), enforces exact phrase matching:
        - "on device" must appear as "on device" or "on-device"
        - NOT "on the measuring device" (words too far apart)

        For single-word terms, uses word boundary matching.

        Args:
            text: Patent title/abstract text
            keyword: Normalized keyword (uppercase, spaces instead of hyphens)

        Returns:
            bool: True if exact phrase found, False otherwise
        """
        if pd.isna(text):
            return False

        text_str = str(text)

        # Use cached compiled pattern for this keyword
        pattern = self.exact_phrase_regex_cache.get(keyword)
        if pattern:
            return bool(pattern.search(text_str))

        # Fallback (shouldn't happen if cache is built correctly)
        return False

    def filter_patent(self, row):
        """
        Decide if patent should be kept.

        Args:
            row: DataFrame row with 'match_term' and 'appln_title'

        Returns:
            tuple: (keep: bool, reason: str, tier: str)
        """
        keyword = row['match_term']
        title = str(row.get('appln_title', ''))

        # FIRST CHECK: Exact phrase matching for multi-word terms
        # This eliminates false positives from SQL wildcard matching (%word1%word2%)
        if not self.exact_phrase_match(title, keyword):
            self.exact_match_rejected += 1
            return False, 'Exact phrase match failed - keyword not found as phrase in title', None

        # Tier 1: Strong keywords (auto-accept)
        if keyword in self.STRONG_KEYWORDS:
            self.strong_count += 1
            return True, 'Strong PET keyword', 'TIER_1_STRONG'

        # Tier 2: Medium keywords (all accepted - track motivation)
        if keyword in self.MEDIUM_KEYWORDS:
            if self.has_privacy_context(title) or self.has_data_ml_context(title):
                self.medium_accepted_privacy += 1
                return True, 'Medium keyword - privacy/data motivated', 'TIER_2_PRIVACY'
            else:
                self.medium_accepted_efficiency += 1
                return True, 'Medium keyword - efficiency motivated', 'TIER_2_EFFICIENCY'

        # Tier 3: Weak keywords (require privacy OR data/ML context)
        if keyword in self.WEAK_KEYWORDS:
            # Check for privacy or data/ML context in title
            has_privacy = self.has_privacy_context(title)
            has_data_ml = self.has_data_ml_context(title)

            # Accept if EITHER privacy context OR data/ML context
            if has_privacy or has_data_ml:
                self.weak_accepted_context += 1
                return True, 'Weak keyword + privacy/ML context in title', 'TIER_3_CONTEXT'
            else:
                self.weak_rejected += 1
                return False, 'Weak keyword without privacy/ML context', 'TIER_3_REJECTED'

        # Unknown keyword - reject (shouldn't happen)
        return False, 'Unknown keyword', 'UNKNOWN'

    def print_summary(self):
        """Print filtering statistics."""
        medium_accepted_total = self.medium_accepted_privacy + self.medium_accepted_efficiency
        weak_accepted_total = self.weak_accepted_context
        total_accepted = self.strong_count + medium_accepted_total + weak_accepted_total
        total_rejected = self.exact_match_rejected + self.weak_rejected
        total = total_accepted + total_rejected

        print("\n" + "="*80)
        print("FILTERING SUMMARY")
        print("="*80)

        print(f"\nACCEPTED: {total_accepted:,} patents")
        print(f"  Tier 1 (Strong keywords):        {self.strong_count:>10,} ({self.strong_count/total*100:>5.1f}%)")
        print(f"  Tier 2 (Privacy-relevant tech):  {medium_accepted_total:>10,} ({medium_accepted_total/total*100:>5.1f}%)")
        print(f"    - Privacy/data motivated:      {self.medium_accepted_privacy:>10,}")
        print(f"    - Efficiency motivated:        {self.medium_accepted_efficiency:>10,}")
        print(f"  Tier 3 (Weak + context):         {weak_accepted_total:>10,} ({weak_accepted_total/total*100:>5.1f}%)")
        print(f"    - Via privacy/ML context:      {self.weak_accepted_context:>10,}")

        print(f"\nREJECTED: {total_rejected:,} patents")
        print(f"  Exact phrase match failed:        {self.exact_match_rejected:>10,} ({self.exact_match_rejected/total*100:>5.1f}%)")
        print(f"  Weak without validation:          {self.weak_rejected:>10,} ({self.weak_rejected/total*100:>5.1f}%)")

        print(f"\nTOTAL PROCESSED: {total:,} patents")
        print(f"Acceptance rate: {total_accepted/total*100:.1f}%")
        print(f"Rejection rate:  {total_rejected/total*100:.1f}%")

        print(f"\n{'='*80}")
        print("EXACT PHRASE MATCHING IMPACT")
        print(f"{'='*80}")
        print(f"  False positives removed: {self.exact_match_rejected:,}")
        print(f"  Primary cause: SQL wildcard matching (%word1%word2%)")
        print(f"  Examples eliminated: 'on THE measuring device', 'data ABOUT minimization'")


def main():
    """Main filtering pipeline."""
    print("="*80)
    print("PET KEYWORD FILTERING PIPELINE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    keyword_df = pd.read_parquet(DATA_DIR / 'pet_ids_keyword_only.parquet')
    print(f"Loaded {len(keyword_df):,} keyword matches")

    # Load titles for context
    print("Loading titles...")
    titles_df = pd.read_parquet(DATA_DIR / 'title_abstract.parquet')

    # Merge with titles
    print("Merging with titles...")
    df = keyword_df.merge(titles_df[['appln_id', 'appln_title']], on='appln_id', how='left')
    print(f"Merged dataset: {len(df):,} rows")

    # Apply filters
    print("\nApplying filters...")
    filter_obj = PETKeywordFilter()

    # Apply filter to each row (OPTIMIZED with progress tracking)
    print(f"Processing {len(df):,} rows (this will take ~5-10 minutes with optimizations)...")
    import time
    start_time = time.time()

    # Process in chunks for progress reporting
    chunk_size = 100000
    results_list = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_results = chunk.apply(filter_obj.filter_patent, axis=1, result_type='expand')
        results_list.append(chunk_results)

        processed = min(i + chunk_size, len(df))
        pct = (processed / len(df)) * 100
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(df) - processed) / rate if rate > 0 else 0

        print(f"  Progress: {processed:,}/{len(df):,} ({pct:.1f}%) | "
              f"{rate:.0f} rows/sec | ETA: {eta/60:.1f} min")

    results = pd.concat(results_list, ignore_index=True)
    results.columns = ['keep', 'reason', 'tier']

    elapsed_total = time.time() - start_time
    print(f"\n✓ Filtering completed in {elapsed_total/60:.1f} minutes ({len(df)/elapsed_total:.0f} rows/sec)")

    df = pd.concat([df, results], axis=1)

    # Print summary
    filter_obj.print_summary()

    # Filter to accepted patents
    df_accepted = df[df['keep']].copy()

    # Remove duplicates: Keep only highest tier per patent
    # Priority: TIER_1 > TIER_2 > TIER_3
    print("\n" + "="*80)
    print("DEDUPLICATION (one patent = one tier)")
    print("="*80)
    print(f"Before deduplication: {len(df_accepted):,} keyword matches")
    print(f"Unique patents: {df_accepted['appln_id'].nunique():,}")

    # Define tier priority (lower number = higher priority)
    tier_priority = {
        'TIER_1_STRONG': 1,
        'TIER_2_PRIVACY': 2,
        'TIER_2_EFFICIENCY': 3,
        'TIER_3_CONTEXT': 4,
    }

    df_accepted['tier_priority'] = df_accepted['tier'].map(tier_priority).fillna(999)

    # Keep only the row with highest priority (lowest number) for each patent
    df_accepted = df_accepted.sort_values('tier_priority').groupby('appln_id').first().reset_index()
    df_accepted = df_accepted.drop('tier_priority', axis=1)

    print(f"After deduplication: {len(df_accepted):,} unique patents")
    print(f"Removed {len(df[df['keep']]) - len(df_accepted):,} duplicate entries")

    # Print tier breakdown after deduplication
    print("\nFinal tier distribution (unique patents only):")
    tier_counts = df_accepted['tier'].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / len(df_accepted) * 100
        print(f"  {tier:<25} {count:>10,} ({pct:>5.1f}%)")

    # Save rejected patents for review (interim data, not final)
    print("\n" + "="*80)
    print("SAVING REJECTED PATENTS (for review)")
    print("="*80)

    df_rejected = df[~df['keep']].copy()
    rejected_file = INTERIM_DIR / 'pet_ids_rejected.parquet'
    df_rejected[['appln_id', 'match_term', 'reason']].to_parquet(rejected_file, index=False)
    print(f"✓ Saved {df_rejected['appln_id'].nunique():,} rejected patents to: {rejected_file}")

    print("\n" + "="*80)
    print("KEYWORD FILTERING COMPLETE")
    print("="*80)
    print(f"Accepted: {df_accepted['appln_id'].nunique():,} patents from keywords")
    print(f"Rejected: {df_rejected['appln_id'].nunique():,} patents")
    print(f"\nNext: Creating NARROW and BROAD definitions...")

    return df_accepted, df_rejected


def create_narrow_and_broad_definitions(df_accepted):
    """
    Create NARROW and BROAD definitions from keyword-filtered baseline.

    NARROW = Keywords (exact phrase, filtered) + RELEVANT_CPC_CODES
      - High precision definition for policy analysis and OECD reporting
      - Low false positives, conservative estimates

    BROAD = NARROW + RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD
      - High recall definition for landscape mapping
      - Comprehensive coverage, may include more noise

    Args:
        df_accepted: DataFrame of keyword-filtered patents (from main())

    Returns:
        tuple: (narrow_df, broad_df) DataFrames ready for analysis
    """
    print("\n" + "="*80)
    print("CREATING NARROW AND BROAD DEFINITIONS")
    print("="*80)

    # Get keyword-based IDs (baseline from filtering)
    keyword_ids = set(df_accepted['appln_id'].unique())
    print(f"\n[1/3] Keyword-filtered baseline: {len(keyword_ids):,} patents")

    # ========================================================================
    # NARROW: Add RELEVANT_CPC_CODES
    # ========================================================================
    print(f"\n[2/3] Creating NARROW definition (OLD + RELEVANT_CPC_CODES)...")

    # Load CPC
    cpc_df = pd.read_parquet(DATA_DIR / 'cpc.parquet')
    cpc_df['cpc_norm'] = cpc_df['cpc_class_symbol'].str.replace(r'\s+', '', regex=True).str.upper()

    # OPTIMIZATION: Build lookup table once (avoids 81,000+ dataframe filters)
    cpc_narrow_code_to_patents = {}
    cpc_narrow_matches = set()

    for code in RELEVANT_CPC_CODES:
        code_norm = code.upper().replace(' ', '')
        matches = set(cpc_df[cpc_df['cpc_norm'].str.startswith(code_norm, na=False)]['appln_id'])
        cpc_narrow_code_to_patents[code] = matches
        cpc_narrow_matches.update(matches)
        print(f"  {code}: {len(matches):,} patents")

    new_from_cpc_narrow = cpc_narrow_matches - keyword_ids
    print(f"\n  Total CPC matches: {len(cpc_narrow_matches):,}")
    print(f"  Already in keyword baseline: {len(keyword_ids & cpc_narrow_matches):,}")
    print(f"  NEW from CPC: {len(new_from_cpc_narrow):,}")

    # Create NARROW dataset
    narrow_records = df_accepted[['appln_id', 'match_term', 'tier', 'reason']].to_dict('records')

    # OPTIMIZATION: Use pre-built lookup instead of filtering dataframe repeatedly
    for appln_id in new_from_cpc_narrow:
        matched_codes = [code for code, patents in cpc_narrow_code_to_patents.items()
                        if appln_id in patents]

        narrow_records.append({
            'appln_id': appln_id,
            'match_term': 'CLASSIFICATION_ONLY',
            'tier': 'TIER_CPC_NARROW',
            'reason': f"Matched RELEVANT_CPC_CODES: {', '.join(matched_codes)}"
        })

    narrow_df = pd.DataFrame(narrow_records)
    narrow_ids = set(narrow_df['appln_id'].unique())

    print(f"\n  NARROW total: {len(narrow_ids):,} patents (+{len(new_from_cpc_narrow):,} vs keyword baseline)")

    # ========================================================================
    # BROAD: Add RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD
    # ========================================================================
    print(f"\n[3/3] Creating BROAD definition (NARROW + IPC/CPC_BROAD)...")

    # Load IPC
    ipc_df = pd.read_parquet(DATA_DIR / 'ipc.parquet')
    ipc_df['ipc_norm'] = ipc_df['ipc_class_symbol'].str.upper().str.strip()

    # OPTIMIZATION: Build IPC lookup table once
    ipc_code_to_patents = {}
    ipc_matches = set()

    for code in RELEVANT_IPC_CODES:
        code_norm = code.upper().replace(' ', '')
        matches1 = set(ipc_df[ipc_df['ipc_norm'].str.startswith(code.upper(), na=False)]['appln_id'])
        matches2 = set(ipc_df[ipc_df['ipc_norm'].str.startswith(code_norm, na=False)]['appln_id'])
        combined = matches1 | matches2
        ipc_code_to_patents[code] = combined
        ipc_matches.update(combined)
        print(f"  IPC {code}: {len(combined):,} patents")

    # OPTIMIZATION: Build CPC_BROAD lookup table once
    cpc_broad_code_to_patents = {}
    cpc_broad_matches = set()

    for code in RELEVANT_CPC_CODES_BROAD:
        code_norm = code.upper().replace(' ', '')
        matches = set(cpc_df[cpc_df['cpc_norm'].str.startswith(code_norm, na=False)]['appln_id'])
        cpc_broad_code_to_patents[code] = matches
        cpc_broad_matches.update(matches)
        print(f"  CPC {code}: {len(matches):,} patents")

    new_from_ipc = ipc_matches - narrow_ids
    new_from_cpc_broad = cpc_broad_matches - narrow_ids

    print(f"\n  Total IPC matches: {len(ipc_matches):,}")
    print(f"  Total CPC_BROAD matches: {len(cpc_broad_matches):,}")
    print(f"  NEW from IPC: {len(new_from_ipc):,}")
    print(f"  NEW from CPC_BROAD: {len(new_from_cpc_broad):,}")

    # Create BROAD dataset
    broad_records = narrow_records.copy()

    # OPTIMIZATION: Use pre-built IPC lookup
    for appln_id in new_from_ipc:
        matched_codes = [code for code, patents in ipc_code_to_patents.items()
                        if appln_id in patents]

        broad_records.append({
            'appln_id': appln_id,
            'match_term': 'CLASSIFICATION_ONLY',
            'tier': 'TIER_IPC_BROAD',
            'reason': f"Matched RELEVANT_IPC_CODES: {', '.join(matched_codes)}"
        })

    # OPTIMIZATION: Use pre-built CPC_BROAD lookup
    for appln_id in (new_from_cpc_broad - new_from_ipc):
        matched_codes = [code for code, patents in cpc_broad_code_to_patents.items()
                        if appln_id in patents]

        broad_records.append({
            'appln_id': appln_id,
            'match_term': 'CLASSIFICATION_ONLY',
            'tier': 'TIER_CPC_BROAD',
            'reason': f"Matched RELEVANT_CPC_CODES_BROAD: {', '.join(matched_codes)}"
        })

    broad_df = pd.DataFrame(broad_records)
    broad_ids = set(broad_df['appln_id'].unique())

    print(f"\n  BROAD total: {len(broad_ids):,} patents (+{len(broad_ids - narrow_ids):,} vs NARROW)")

    # ========================================================================
    # Save NARROW and BROAD to GOLD directory (final, blessed datasets)
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING FINAL DATASETS TO GOLD DIRECTORY")
    print("="*80)

    narrow_file = GOLD_DIR / 'pet_ids_narrow.parquet'
    narrow_df.to_parquet(narrow_file, index=False)
    print(f"\n✓ NARROW: {narrow_file}")
    print(f"  {len(narrow_ids):,} patents (high precision)")

    broad_file = GOLD_DIR / 'pet_ids_broad.parquet'
    broad_df.to_parquet(broad_file, index=False)
    print(f"\n✓ BROAD: {broad_file}")
    print(f"  {len(broad_ids):,} patents (high recall)")

    # Save summary to interim (metadata, not final dataset)
    summary = pd.DataFrame([
        {'definition': 'NARROW', 'patents': len(narrow_ids), 'description': 'Keywords (exact phrase) + RELEVANT_CPC_CODES'},
        {'definition': 'BROAD', 'patents': len(broad_ids), 'description': 'NARROW + RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD'}
    ])
    summary_file = INTERIM_DIR / 'definitions_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\n✓ Summary: {summary_file}")

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE - GOLD DATASETS READY")
    print("="*80)
    print(f"\n  Keywords (filtered):  {len(keyword_ids):>8,} patents")
    print(f"  NARROW (+ CPC):       {len(narrow_ids):>8,} patents (+{len(narrow_ids - keyword_ids):>6,})")
    print(f"  BROAD (+ IPC/CPC):    {len(broad_ids):>8,} patents (+{len(broad_ids - narrow_ids):>6,})")

    print(f"\n  📊 Use NARROW for: Precise statistics")
    print(f"  📊 Use BROAD for: Technology landscape, comprehensive coverage")
    print(f"\n  ⚠️  IMPORTANT: All analysis should use BOTH definitions to show robustness")

    print(f"\n  Next steps:")
    print(f"    1. Run analysis scripts with both definitions")
    print(f"    2. Compare results (should be similar for robust findings)")
    print(f"    3. See: MIGRATION_QUICK_START.md for usage examples")
    print()

    return narrow_df, broad_df


if __name__ == '__main__':
    df_accepted, df_rejected = main()
    narrow_df, broad_df = create_narrow_and_broad_definitions(df_accepted)
