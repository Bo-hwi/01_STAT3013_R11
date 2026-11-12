# M4 controlled feature reconciliation

Adapter: `python -B -m src.decision.feature_staging` (project `.venv`).
This command builds features only; it never fits a forecasting model or adopts outputs.

## Contract and metadata

The unmodified `src/features/lag_features.py::build_features()` creates the ordered
34-column contract: 23 `PASSTHROUGH`, four `SALES_LAGS`, two `SALES_ROLLING`, four
`WEATHER_LAG1`, and `score_mask`. `excluded_post_reopening` stays internal to M1.
Raw per-city weather is required to preserve calendar lags around Christmas.

M1 `write_metadata()` has a fixed canonical destination. The adapter inspects its
AST and evaluates only the actual metadata dictionary expression in a copied
namespace, with the candidate DataFrame. It never compiles the write statement,
calls the writer, or changes module globals. Unexpected writer structure fails
before staging. A synthetic test intercepts the real writer's IO and compares
its serialized metadata with the adapter output. Provenance additions live in
the reconciliation report, leaving the M1 JSON contract unchanged.

## Safety and evidence

`regenerate_candidate()` requires a dedicated directory inside `data/processed/`.
Each run exclusively creates a unique `m4_feature_candidates/candidate-*` directory
and exclusively creates every file. It copies curated Parquet and metadata into
that directory and verifies SHA-256 equality, enabling the unchanged production
preflight's sibling-path interface. These are independent copies, not links.
Canonical Parquet, metadata and raw files are never written or replaced.

Comparison aligns exact `(date, store_nbr, family)` keys, then uses exact pandas
series equality on all 34 columns, including dtypes, categorical metadata and NaN
positions. No rounding or numerical tolerance hides differences. Existing-only
columns and all metadata differences are reported. The unchanged
`production_preflight()` checks the candidate plus its staged curated evidence
against the original catalog, raw weather, configs and baseline count reference.

`reconciliation_report.json` records before/after source hashes, builder/adapter/
preflight/dependency/config hashes, dependency versions, ordered schema, column
comparisons, preflight evidence, output hashes, actual elapsed times and optional
process memory/CPU measurements. Peak working set is process-lifetime, not an
isolated builder increment. `CANDIDATE_COMPLETE.json` binds the report digest and
states `adopted: false`; it means reconciliation finished, not that preflight
passed. Always inspect `recommendation` and `preflight_status`. Failed execution
leaves partial files and `FAILED.json` where writable, without a completion marker.

`GO_FOR_ADOPTION_REVIEW` requires all shared columns exact, only the expected
extra helper in the old file, and passing production preflight. It authorizes
neither replacing canonical artifacts nor model training. Explicit user approval
is required before any adoption. Existing preflight warnings/unavailable upstream
evidence retain their meaning, including missing approved processed-file digests,
retrospective ERA5 availability and validation-count references.

Synthetic tests: `tests/test_m4_feature_staging.py`; they do not invoke forecasting
training. No synthetic results constitute research evidence.
