"""Batch conversion of a whole historical population table.

The converter accepts a v1 handoff package plus a UTF-8 CSV whose first
column is ``row_id`` and whose remaining columns correspond one-to-one to
the old district codes in the package.  Blank cells mean "not reported";
``0`` is a real reported value.  The table is streamed row by row — memory
use does not grow with the number of rows — and the converted CSV is
spooled to a temporary file so the whole batch is all-or-nothing: only
after every record passes validation and exact conversion is the CSV
offered for download.

Conversion rules are prepared once per distinct reported-district
combination (the count is returned in the ``X-Rule-Preparations`` response
header); every row is still evaluated with its own non-negative integer
populations under exactly the same semantics as ``POST /convert``.
"""
import csv
import io
import json
import re
import tempfile
from typing import Dict, Iterator, List, Tuple

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .engine import ConversionError, PreparedRules, evaluate_rules, prepare_rules
from .models import HandoffPackage

# Batch contract limits.
MAX_ROWS = 100_000
MAX_OLD_DISTRICTS = 40
MAX_NEW_DISTRICTS = 40
MAX_REPORTED_COMBINATIONS = 8

MAX_PACKAGE_BYTES = 1 << 20  # a v1 package with <=40+<=40 districts is tiny
_OUTPUT_MEMORY_BYTES = 1 << 20  # beyond this the output CSV spools to disk
_CHUNK_SIZE = 64 * 1024

RULE_PREPARATIONS_HEADER = "X-Rule-Preparations"

_INT_RE = re.compile(r"[0-9]+")


class BatchInputError(ValueError):
    """Raised when the uploaded package or CSV table is not compliant."""


def _load_package(upload: UploadFile) -> dict:
    """Read, parse and validate the uploaded v1 handoff package."""
    raw = upload.file.read(MAX_PACKAGE_BYTES + 1)
    if len(raw) > MAX_PACKAGE_BYTES:
        raise BatchInputError(
            f"package file exceeds the {MAX_PACKAGE_BYTES} byte limit"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchInputError("package file is not valid UTF-8") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchInputError(f"package file is not valid JSON: {exc}") from exc
    try:
        model = HandoffPackage.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise BatchInputError(f"invalid handoff package: {problems}") from exc
    if model.schema_version != 1:
        raise BatchInputError(
            f"unsupported schema_version {model.schema_version}: "
            "only v1 handoff packages are accepted"
        )
    if len(model.old_districts) > MAX_OLD_DISTRICTS:
        raise BatchInputError(
            f"package has {len(model.old_districts)} old districts, "
            f"at most {MAX_OLD_DISTRICTS} are supported"
        )
    if len(model.new_districts) > MAX_NEW_DISTRICTS:
        raise BatchInputError(
            f"package has {len(model.new_districts)} new districts, "
            f"at most {MAX_NEW_DISTRICTS} are supported"
        )
    return model.model_dump()


def _read_header(reader, old_codes: List[str]) -> List[int]:
    """Validate the CSV header; return csv column -> old-district index."""
    header = next(reader, None)
    if header is None:
        raise BatchInputError("CSV table is empty")
    if not header or header[0] != "row_id":
        raise BatchInputError('first CSV column must be "row_id"')
    columns = header[1:]
    if len(columns) != len(old_codes):
        raise BatchInputError(
            "CSV header must list every old district of the package: "
            f"expected {len(old_codes)} column(s) after row_id, "
            f"got {len(columns)}"
        )
    if len(set(columns)) != len(columns):
        raise BatchInputError("duplicate old district column in CSV header")
    unknown = sorted(set(columns) - set(old_codes))
    if unknown:
        raise BatchInputError(
            f"CSV columns not among the package old districts: {unknown}"
        )
    missing = sorted(set(old_codes) - set(columns))
    if missing:
        raise BatchInputError(f"CSV header misses old districts: {missing}")
    return [old_codes.index(name) for name in columns]


def _iter_spool(spool, closer) -> Iterator[bytes]:
    """Stream the spooled output CSV, then release the temporary file."""
    try:
        while True:
            chunk = spool.read(_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        closer.close()


class BatchRunner:
    """Runs one batch job; ``preparations`` counts prepared rule sets."""

    def __init__(self) -> None:
        self.preparations = 0

    def run(self, package: UploadFile, table: UploadFile) -> StreamingResponse:
        pkg = _load_package(package)
        old_codes: List[str] = pkg["old_districts"]
        new_codes: List[str] = pkg["new_districts"]
        n_old = len(old_codes)
        width = 1 + n_old

        reader = csv.reader(
            io.TextIOWrapper(table.file, encoding="utf-8-sig", newline="")
        )
        column_index = _read_header(reader, old_codes)

        spool = tempfile.SpooledTemporaryFile(
            max_size=_OUTPUT_MEMORY_BYTES, mode="w+b"
        )
        out_text = io.TextIOWrapper(spool, encoding="utf-8", newline="")
        writer = csv.writer(out_text)
        try:
            writer.writerow(["row_id", *new_codes])
            cache: Dict[Tuple[int, ...], PreparedRules] = {}
            record_no = 0
            for fields in reader:
                record_no += 1
                if record_no > MAX_ROWS:
                    raise BatchInputError(
                        f"record {record_no}: row count exceeds the "
                        f"{MAX_ROWS} row limit"
                    )
                if len(fields) != width:
                    raise BatchInputError(
                        f"record {record_no}: expected {width} fields, "
                        f"got {len(fields)}"
                    )
                row_id = fields[0]
                populations = [0] * n_old
                reported: List[int] = []
                for col, cell in enumerate(fields[1:]):
                    value = cell.strip()
                    if not value:
                        continue  # blank means "not reported"
                    if not _INT_RE.fullmatch(value):
                        name = old_codes[column_index[col]]
                        raise BatchInputError(
                            f"record {record_no} (row_id {row_id!r}): value "
                            f"for old district {name!r} must be blank (not "
                            f"reported) or a non-negative integer, "
                            f"got {cell!r}"
                        )
                    idx = column_index[col]
                    populations[idx] = int(value)
                    reported.append(idx)
                key = tuple(sorted(reported))
                rules = cache.get(key)
                if rules is None:
                    if len(cache) >= MAX_REPORTED_COMBINATIONS:
                        raise BatchInputError(
                            f"record {record_no}: distinct reported-district "
                            f"combinations exceed {MAX_REPORTED_COMBINATIONS}"
                        )
                    rules = prepare_rules(pkg, [old_codes[i] for i in key])
                    cache[key] = rules
                    self.preparations += 1
                try:
                    values = evaluate_rules(rules, populations)
                except ConversionError as exc:
                    raise ConversionError(
                        f"record {record_no} (row_id {row_id!r}): {exc}"
                    ) from exc
                writer.writerow(
                    [row_id] + ["" if v is None else v for v in values]
                )
            out_text.flush()
        except Exception:
            out_text.close()  # also releases the spooled temporary file
            raise

        spool.seek(0)
        return StreamingResponse(
            _iter_spool(spool, out_text),
            media_type="text/csv",
            headers={
                RULE_PREPARATIONS_HEADER: str(self.preparations),
                "Content-Disposition": 'attachment; filename="converted.csv"',
            },
        )
