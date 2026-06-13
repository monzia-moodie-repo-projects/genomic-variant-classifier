from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Optional

import pandas as pd



_STRING_DTYPE_FAMILY = frozenset({
    "object", "string", "str",
    "string[python]", "string[pyarrow]", "string[pyarrow_numpy]",
    "large_string[pyarrow]",
})


def _dtype_family(dtype: str) -> str:
    """Collapse equivalent string-dtype spellings to one family token.

    pandas 3.0 defaults the inferred-string dtype to 'str'/'string' where 2.x used
    'object'; treating those as one family keeps a string column from registering as
    drift purely because of the pandas version. IDENTITY for every other dtype, so
    genuine retyping (float64 -> int64) is still caught and numeric-only baselines
    hash identically (no rebuild)."""
    if str(dtype).strip().lower() in _STRING_DTYPE_FAMILY:
        return "string"
    return str(dtype)


@dataclass(frozen=True)
class SchemaDriftResult:
    timestamp: str
    expected_schema_hash: str
    observed_schema_hash: str
    columns_added: tuple[str, ...]
    columns_removed: tuple[str, ...]
    columns_dtype_changed: tuple[tuple[str, str, str], ...]  # (col, expected, observed)
    pandera_violations: tuple[str, ...]
    severity: str  # green | red (no amber — schema drift is binary)


@dataclass
class SchemaDriftAgent:
    """Pandera-based schema-contract enforcement on the feature matrix.

    Acts as a hard gate at the Spark ETL boundary. Schema drift is treated
    as red severity: any change halts ETL and emits a hypothesis stub.
    """

    schema: pa.DataFrameSchema
    expected_dtypes: dict[str, str]
    expected_schema_hash: str
    output_dir: Path
    logger: Optional[Logger] = field(default=None, repr=False)

    @staticmethod
    def hash_schema(dtypes: dict[str, str]) -> str:
        canonical = json.dumps(
            sorted((str(k), _dtype_family(v)) for k, v in dtypes.items()),
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_baseline(cls, baseline_path, output_dir):
        """Reconstruct a detector from a schema-baseline JSON.

        The pandera schema is rebuilt from expected_dtypes with nullable columns so that
        degenerate-but-present (all-NaN) columns do not raise false nullability violations
        against their own baseline. pandera is imported lazily (optional dep).
        """
        import pandera.pandas as pa
        data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        expected_dtypes = {str(k): str(v) for k, v in data["expected_dtypes"].items()}
        schema = pa.DataFrameSchema(
            {
                col: pa.Column(
                    None if _dtype_family(dtype) == "string" else dtype,
                    nullable=True,
                )
                for col, dtype in expected_dtypes.items()
            }
        )
        return cls(
            schema=schema,
            expected_dtypes=expected_dtypes,
            expected_schema_hash=data["expected_schema_hash"],
            output_dir=Path(output_dir),
        )

    def detect(self, df: pd.DataFrame) -> SchemaDriftResult:
        import pandera.pandas as pa  # lazy: required only when detection runs
        observed_dtypes = {c: str(df[c].dtype) for c in df.columns}
        observed_hash = self.hash_schema(observed_dtypes)
        expected_cols = set(self.expected_dtypes)
        observed_cols = set(observed_dtypes)
        added = tuple(sorted(observed_cols - expected_cols))
        removed = tuple(sorted(expected_cols - observed_cols))
        changed: list[tuple[str, str, str]] = []
        for col in expected_cols & observed_cols:
            if _dtype_family(observed_dtypes[col]) != _dtype_family(self.expected_dtypes[col]):
                changed.append((col, self.expected_dtypes[col], observed_dtypes[col]))
        violations: list[str] = []
        try:
            self.schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:  # pragma: no cover - exercised in tests
            violations = [str(e) for e in exc.failure_cases.itertuples(index=False)]
        clean = (
            observed_hash == self.expected_schema_hash
            and not added
            and not removed
            and not changed
            and not violations
        )
        return SchemaDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            expected_schema_hash=self.expected_schema_hash,
            observed_schema_hash=observed_hash,
            columns_added=added,
            columns_removed=removed,
            columns_dtype_changed=tuple(changed),
            pandera_violations=tuple(violations),
            severity="green" if clean else "red",
        )

    def persist(self, result: SchemaDriftResult, run_id: str) -> Path:
        out = self.output_dir / "schema" / f"{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result.__dict__, default=str, indent=2), encoding="utf-8"
        )
        return out
