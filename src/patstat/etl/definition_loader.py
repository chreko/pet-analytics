"""
Helper module to load PET patent definitions consistently across all scripts.

This enforces the dual-definition approach required by proposed methodology.

IMPORTANT: All analysis scripts should use BOTH narrow and broad definitions
to demonstrate robustness of findings across different precision/recall tradeoffs.

DEFINITIONS:
-----------
NARROW = Keywords (exact phrase, filtered) + RELEVANT_CPC_CODES
  - High precision, low false positives
  - Use for: Precise statistics

BROAD = NARROW + RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD
  - High recall, comprehensive coverage
  - Use for: Technology landscape, comprehensive analysis, emerging trends

LOCATION:
--------
Definitions are stored in data/gold/ (final, blessed datasets):
  - data/gold/pet_ids_narrow.parquet
  - data/gold/pet_ids_broad.parquet

USAGE:
-----
    from patstat.etl.definition_loader import load_definitions

    # Load both (recommended for analysis)
    narrow_df, broad_df = load_definitions(which='both')

    # Load one definition
    narrow_df = load_definitions(which='narrow')
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, Literal, Optional
import warnings

DEFAULT_GOLD_DIR = Path(__file__).parent.parent / 'data' / 'gold'


def load_definitions(
    gold_dir: Optional[Path] = None,
    which: Literal['both', 'narrow', 'broad'] = 'both'
) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """
    Load PET patent definitions from gold directory.

    Args:
        gold_dir: Path to gold data directory (uses default if None)
        which: Which definition(s) to load
            - 'both': Returns (narrow_df, broad_df) tuple
            - 'narrow': Returns narrow_df only (high precision)
            - 'broad': Returns broad_df only (high recall)

    Returns:
        DataFrame or tuple of DataFrames with columns:
        - appln_id: Patent application ID
        - match_term: Keyword or classification code that matched
        - tier: Tier classification (TIER_1_STRONG, TIER_CPC_NARROW, etc.)
        - reason: Why this patent was included

    Examples:
        >>> # Load both definitions (recommended for analysis)
        >>> narrow_df, broad_df = load_definitions(which='both')
        >>>
        >>> # Load only narrow (for high-precision sampling)
        >>> narrow_df = load_definitions(which='narrow')
        >>>
        >>> # Get patent ID sets
        >>> narrow_ids = set(narrow_df['appln_id'].unique())
        >>> broad_ids = set(broad_df['appln_id'].unique())
    """
    if gold_dir is None:
        gold_dir = DEFAULT_GOLD_DIR

    narrow_file = gold_dir / 'pet_ids_narrow.parquet'
    broad_file = gold_dir / 'pet_ids_broad.parquet'

    # Verify files exist
    if not narrow_file.exists():
        raise FileNotFoundError(
            f"NARROW definition not found: {narrow_file}\n\n"
            f"Run the following to generate:\n"
            f"  python src/patstat/etl/transform_filter_patents.py\n"
        )

    if not broad_file.exists():
        raise FileNotFoundError(
            f"BROAD definition not found: {broad_file}\n\n"
            f"Run the following to generate:\n"
            f"  python src/patstat/etl/transform_filter_patents.py\n"
        )

    if which == 'narrow':
        return pd.read_parquet(narrow_file)
    elif which == 'broad':
        return pd.read_parquet(broad_file)
    elif which == 'both':
        return pd.read_parquet(narrow_file), pd.read_parquet(broad_file)
    else:
        raise ValueError(
            f"Invalid 'which' parameter: {which}. "
            f"Must be one of: 'both', 'narrow', 'broad'"
        )


def get_definition_ids(
    gold_dir: Optional[Path] = None,
    which: Literal['narrow', 'broad'] = 'narrow'
) -> set:
    """
    Get set of patent IDs for a definition (fast version).

    This is more efficient than loading the full dataframe when you only need IDs.

    Args:
        gold_dir: Path to gold data directory (uses default if None)
        which: Which definition ('narrow' or 'broad')

    Returns:
        Set of appln_id values

    Examples:
        >>> narrow_ids = get_definition_ids(which='narrow')
        >>> broad_ids = get_definition_ids(which='broad')
        >>> new_in_broad = broad_ids - narrow_ids
        >>> print(f"BROAD adds {len(new_in_broad):,} patents vs NARROW")
    """
    df = load_definitions(gold_dir, which=which)
    return set(df['appln_id'].unique())


def print_definition_summary(gold_dir: Optional[Path] = None):
    """
    Print summary statistics for both definitions.

    Useful for quick sanity checks and understanding the scope of each definition.

    Args:
        gold_dir: Path to gold data directory (uses default if None)

    Examples:
        >>> from patstat.etl.definition_loader import print_definition_summary
        >>> print_definition_summary()
    """
    if gold_dir is None:
        gold_dir = DEFAULT_GOLD_DIR

    narrow_df, broad_df = load_definitions(gold_dir, which='both')

    narrow_ids = set(narrow_df['appln_id'].unique())
    broad_ids = set(broad_df['appln_id'].unique())

    print("="*80)
    print("PET PATENT DEFINITIONS SUMMARY")
    print("="*80)
    print(f"\nNARROW: {len(narrow_ids):>8,} unique patents")
    print(f"  - High precision definition")
    print(f"  - Keywords (exact phrase) + RELEVANT_CPC_CODES")
    print(f"  - Use for: Policy analysis, precise statistics")

    print(f"\nBROAD:  {len(broad_ids):>8,} unique patents")
    print(f"  - High recall definition")
    print(f"  - NARROW + RELEVANT_IPC_CODES + RELEVANT_CPC_CODES_BROAD")
    print(f"  - Use for: Technology landscape mapping, comprehensive coverage")

    print(f"\nDifference: {len(broad_ids - narrow_ids):>8,} patents")
    print(f"  - {(len(broad_ids - narrow_ids) / len(broad_ids) * 100):.1f}% of BROAD")
    print(f"  - {(len(narrow_ids) / len(broad_ids) * 100):.1f}% overlap")

    print("\n" + "="*80)
    print("TIER DISTRIBUTIONS")
    print("="*80)

    print("\nNARROW tiers:")
    for tier, count in narrow_df['tier'].value_counts().sort_index().items():
        pct = count / len(narrow_df) * 100
        print(f"  {tier:<25} {count:>10,} ({pct:>5.1f}%)")

    print("\nBROAD tiers:")
    for tier, count in broad_df['tier'].value_counts().sort_index().items():
        pct = count / len(broad_df) * 100
        print(f"  {tier:<25} {count:>10,} ({pct:>5.1f}%)")

    print()


def compare_definitions(gold_dir: Optional[Path] = None):
    """
    Compare narrow and broad definitions in detail.

    Returns statistics about overlap, unique patents, and tier composition.

    Args:
        gold_dir: Path to gold data directory (uses default if None)

    Returns:
        dict: Comparison statistics

    Examples:
        >>> stats = compare_definitions()
        >>> print(f"Overlap: {stats['overlap_pct']:.1f}%")
    """
    if gold_dir is None:
        gold_dir = DEFAULT_GOLD_DIR

    narrow_df, broad_df = load_definitions(gold_dir, which='both')

    narrow_ids = set(narrow_df['appln_id'].unique())
    broad_ids = set(broad_df['appln_id'].unique())

    overlap = narrow_ids & broad_ids
    only_narrow = narrow_ids - broad_ids
    only_broad = broad_ids - narrow_ids

    return {
        'narrow_total': len(narrow_ids),
        'broad_total': len(broad_ids),
        'overlap': len(overlap),
        'only_narrow': len(only_narrow),
        'only_broad': len(only_broad),
        'overlap_pct': len(overlap) / len(broad_ids) * 100 if broad_ids else 0,
        'narrow_tiers': narrow_df['tier'].value_counts().to_dict(),
        'broad_tiers': broad_df['tier'].value_counts().to_dict(),
    }


# =============================================================================
# LEGACY COMPATIBILITY (DEPRECATED)
# =============================================================================

def load_filtered(gold_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    DEPRECATED: Load legacy filtered definition.

    This function exists for backward compatibility only.
    New code should use load_definitions(which='narrow') instead.

    The 'filtered' definition is now called 'narrow' to better reflect
    its role in the dual-definition approach (narrow = high precision,
    broad = high recall).

    Will be removed in a future version.

    Args:
        gold_dir: Path to gold data directory (uses default if None)

    Returns:
        DataFrame: Narrow definition (same as old 'filtered')

    Examples:
        >>> # OLD (deprecated)
        >>> filtered_df = load_filtered()
        >>>
        >>> # NEW (preferred)
        >>> narrow_df = load_definitions(which='narrow')
    """
    warnings.warn(
        "load_filtered() is deprecated and will be removed in a future version.\n"
        "Use load_definitions(which='narrow') instead.\n"
        "See: MIGRATION_QUICK_START.md",
        DeprecationWarning,
        stacklevel=2
    )
    return load_definitions(gold_dir, which='narrow')


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    import sys

    print("Testing definition_loader module...")
    print()

    # Test 1: Print summary
    print("TEST 1: Printing definition summary")
    print("-" * 80)
    try:
        print_definition_summary()
        print("✓ Summary printed successfully")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print()

    # Test 2: Load both
    print("TEST 2: Loading both definitions")
    print("-" * 80)
    try:
        narrow_df, broad_df = load_definitions(which='both')
        print(f"✓ NARROW: {len(narrow_df):,} records, {narrow_df['appln_id'].nunique():,} unique patents")
        print(f"✓ BROAD: {len(broad_df):,} records, {broad_df['appln_id'].nunique():,} unique patents")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print()

    # Test 3: Get IDs
    print("TEST 3: Getting patent ID sets")
    print("-" * 80)
    try:
        narrow_ids = get_definition_ids(which='narrow')
        broad_ids = get_definition_ids(which='broad')
        print(f"✓ NARROW: {len(narrow_ids):,} unique IDs")
        print(f"✓ BROAD: {len(broad_ids):,} unique IDs")
        print(f"✓ BROAD adds {len(broad_ids - narrow_ids):,} patents vs NARROW")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print()

    # Test 4: Compare
    print("TEST 4: Comparing definitions")
    print("-" * 80)
    try:
        stats = compare_definitions()
        print(f"✓ Overlap: {stats['overlap']:,} patents ({stats['overlap_pct']:.1f}%)")
        print(f"✓ Only NARROW: {stats['only_narrow']:,} patents")
        print(f"✓ Only BROAD: {stats['only_broad']:,} patents")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print()

    # Test 5: Deprecation warning
    print("TEST 5: Testing deprecation warning")
    print("-" * 80)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = load_filtered()
            if len(w) == 1 and issubclass(w[0].category, DeprecationWarning):
                print("✓ Deprecation warning raised correctly")
            else:
                print("✗ Deprecation warning not raised")
    except Exception as e:
        print(f"✗ Error: {e}")

    print()
    print("="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
