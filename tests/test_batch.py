"""End-to-end tests for the batch CSV conversion endpoint."""
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from app import batch
from app.builder_app import app as builder_app
from app.converter_app import app as converter_app

builder = TestClient(builder_app)
converter = TestClient(converter_app)


def build(old, new):
    resp = builder.post(
        "/build", json={"old_districts": old, "new_districts": new}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def simple_package():
    # Disjoint old districts; N1 = O1 + O2, N2 = O3.
    return build(
        [
            {"code": "O1", "blocks": ["b1"]},
            {"code": "O2", "blocks": ["b2"]},
            {"code": "O3", "blocks": ["b3"]},
        ],
        [
            {"code": "N1", "blocks": ["b1", "b2"]},
            {"code": "N2", "blocks": ["b3"]},
        ],
    )


def make_csv(header, rows) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def post_batch(package, csv_bytes=None, package_bytes=None):
    if package_bytes is None:
        package_bytes = json.dumps(package).encode("utf-8")
    files = {"package": ("package.json", package_bytes, "application/json")}
    if csv_bytes is not None:
        files["table"] = ("table.csv", csv_bytes, "text/csv")
    return converter.post("/convert/batch", files=files)


def read_csv(text):
    return list(csv.reader(io.StringIO(text)))


# --------------------------------------------------------------------------
# Happy path: whole table converted, layout and headers as specified.
# --------------------------------------------------------------------------
def test_batch_converts_whole_table():
    pkg = simple_package()
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3"],
        [
            ["r1", "10", "5", "7"],
            ["r2", "0", "0", "0"],   # zeros are real reports, not blanks
            ["r3", "4", "", "9"],    # O2 not reported -> N1 underivable
            ["r4", "", "", "3"],     # only O3 reported
        ],
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    # distinct combinations: {O1,O2,O3} x2, {O1,O3}, {O3} -> 3 preparations
    assert resp.headers["x-rule-preparations"] == "3"
    rows = read_csv(resp.text)
    assert rows[0] == ["row_id", "N1", "N2"]  # new districts, package order
    assert rows[1] == ["r1", "15", "7"]
    assert rows[2] == ["r2", "0", "0"]
    assert rows[3] == ["r3", "", "9"]
    assert rows[4] == ["r4", "", "3"]


def test_batch_prepares_rules_once_per_combination():
    pkg = simple_package()
    rows = [["r1", "1", "2", "3"], ["r2", "4", "5", "6"], ["r3", "7", "8", "9"]]
    resp = post_batch(pkg, make_csv(["row_id", "O1", "O2", "O3"], rows))
    assert resp.status_code == 200
    assert resp.headers["x-rule-preparations"] == "1"
    assert [r[1] for r in read_csv(resp.text)[1:]] == ["3", "9", "15"]


def test_batch_columns_matched_by_name_not_position():
    pkg = simple_package()
    csv_data = make_csv(["row_id", "O3", "O1", "O2"], [["r1", "7", "10", "5"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 200
    assert read_csv(resp.text)[1] == ["r1", "15", "7"]


def test_batch_whitespace_padded_cells():
    pkg = simple_package()
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3"], [["r1", " 10 ", " ", "7"]]
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 200
    # " 10 " parses as 10, whitespace-only cell counts as not reported.
    assert read_csv(resp.text)[1] == ["r1", "", "7"]


def test_batch_underivable_new_district_column_stays_blank():
    pkg = build(
        [{"code": "O1", "blocks": ["a"]}],
        [
            {"code": "N1", "blocks": ["a"]},
            {"code": "N2", "blocks": ["b"]},  # never covered by old districts
        ],
    )
    resp = post_batch(pkg, make_csv(["row_id", "O1"], [["r1", "5"]]))
    assert resp.status_code == 200
    assert read_csv(resp.text) == [["row_id", "N1", "N2"], ["r1", "5", ""]]


def test_batch_utf8_bom_and_chinese_codes():
    pkg = build(
        [
            {"code": "旧甲", "blocks": ["a"]},
            {"code": "旧乙", "blocks": ["b"]},
        ],
        [{"code": "新合", "blocks": ["a", "b"]}],
    )
    csv_data = b"\xef\xbb\xbf" + make_csv(
        ["row_id", "旧甲", "旧乙"], [["记录一", "3", "4"]]
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 200, resp.text
    rows = read_csv(resp.text)
    assert rows[0] == ["row_id", "新合"]
    assert rows[1] == ["记录一", "7"]


def test_batch_supports_100k_rows():
    pkg = build(
        [{"code": "O1", "blocks": ["a"]}],
        [{"code": "N1", "blocks": ["a"]}],
    )
    lines = ["row_id,O1"] + [f"r{i},{i}" for i in range(100_000)]
    resp = post_batch(pkg, "\n".join(lines).encode("utf-8"))
    assert resp.status_code == 200
    assert resp.headers["x-rule-preparations"] == "1"
    rows = resp.text.splitlines()
    assert len(rows) == 100_001
    assert rows[0] == "row_id,N1"
    assert rows[1] == "r0,0"
    assert rows[-1] == "r99999,99999"


# --------------------------------------------------------------------------
# Conversion semantics identical to /convert, errors name the record.
# --------------------------------------------------------------------------
def test_batch_relation_violation_names_record():
    # Identical coverage -> exact relation O1 - O2 = 0.
    pkg = build(
        [{"code": "O1", "blocks": ["x"]}, {"code": "O2", "blocks": ["x"]}],
        [{"code": "N1", "blocks": ["x"]}],
    )
    csv_data = make_csv(
        ["row_id", "O1", "O2"], [["r1", "3", "3"], ["r2", "3", "4"]]
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "conversion_failed"
    assert "record 2" in body["detail"]
    assert "r2" in body["detail"]
    assert "coverage relation" in body["detail"]
    # one rule set was prepared before the failure
    assert resp.headers["x-rule-preparations"] == "1"


def test_batch_negative_result_names_record():
    pkg = build(
        [
            {"code": "P1", "blocks": ["y1", "y2"]},
            {"code": "P2", "blocks": ["y2"]},
        ],
        [{"code": "M1", "blocks": ["y1"]}],
    )
    csv_data = make_csv(["row_id", "P1", "P2"], [["r1", "3", "10"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "conversion_failed"
    assert "record 1" in body["detail"]
    assert "negative" in body["detail"]


def test_batch_non_integer_result_names_record():
    # N1 = 1/2*O1 - 1/2*O3 + 1/2*O4 (see the JSON API tests).
    pkg = build(
        [
            {"code": "O1", "blocks": ["x3", "x4"]},
            {"code": "O2", "blocks": ["x3", "x4"]},
            {"code": "O3", "blocks": ["x1", "x4"]},
            {"code": "O4", "blocks": ["x1", "x3"]},
        ],
        [{"code": "N1", "blocks": ["x3"]}],
    )
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3", "O4"],
        [["r1", "5", "5", "4", "3"], ["r2", "6", "6", "4", "3"]],
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "conversion_failed"
    assert "record 2" in body["detail"]
    assert "non-integer" in body["detail"]


def test_batch_fractional_coefficients_convert_exactly():
    pkg = build(
        [
            {"code": "O1", "blocks": ["x3", "x4"]},
            {"code": "O2", "blocks": ["x3", "x4"]},
            {"code": "O3", "blocks": ["x1", "x4"]},
            {"code": "O4", "blocks": ["x1", "x3"]},
        ],
        [{"code": "N1", "blocks": ["x3"]}],
    )
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3", "O4"], [["r1", "5", "5", "4", "3"]]
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 200
    assert read_csv(resp.text)[1] == ["r1", "2"]


# --------------------------------------------------------------------------
# Input compliance: only an error body is returned.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["-1", "1.5", "abc", "1e3", "+7", "1,000"])
def test_batch_rejects_non_integer_cells(bad):
    pkg = simple_package()
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3"],
        [["r1", "1", "2", "3"], ["r2", bad, "2", "3"]],
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert set(body) == {"error", "detail"}
    assert body["error"] == "invalid_input"
    assert "record 2" in body["detail"]
    assert "r2" in body["detail"]


def test_batch_header_must_start_with_row_id():
    pkg = simple_package()
    csv_data = make_csv(["id", "O1", "O2", "O3"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"
    assert "row_id" in resp.json()["detail"]


def test_batch_header_missing_district_column():
    pkg = simple_package()
    csv_data = make_csv(["row_id", "O1", "O2"], [["r1", "1", "2"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert "expected 3" in resp.json()["detail"]


def test_batch_header_unknown_district_column():
    pkg = simple_package()
    csv_data = make_csv(["row_id", "O1", "O2", "ZZ"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert "ZZ" in resp.json()["detail"]


def test_batch_header_duplicate_district_column():
    pkg = simple_package()
    csv_data = make_csv(["row_id", "O1", "O1", "O2"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert "duplicate" in resp.json()["detail"]


def test_batch_ragged_row_rejected():
    pkg = simple_package()
    resp = post_batch(pkg, b"row_id,O1,O2,O3\nr1,1,2\n")
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "invalid_input"
    assert "record 1" in body["detail"]
    assert "expected 4 fields" in body["detail"]


def test_batch_empty_csv_rejected():
    pkg = simple_package()
    resp = post_batch(pkg, b"")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_batch_rejects_non_v1_package():
    pkg = simple_package()
    pkg["schema_version"] = 2
    csv_data = make_csv(["row_id", "O1", "O2", "O3"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"
    assert "schema_version" in resp.json()["detail"]


def test_batch_rejects_malformed_package_json():
    pkg = simple_package()
    csv_data = make_csv(["row_id", "O1", "O2", "O3"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data, package_bytes=b"{not json")
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_batch_rejects_tampered_package():
    pkg = simple_package()
    pkg["old_relations"] = [{"ZZ": 1}]
    csv_data = make_csv(["row_id", "O1", "O2", "O3"], [["r1", "1", "2", "3"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"


def test_batch_rejects_more_than_40_old_districts():
    old = [{"code": f"O{i}", "blocks": [f"b{i}"]} for i in range(41)]
    pkg = build(old, [{"code": "N1", "blocks": ["b0"]}])
    header = ["row_id"] + [d["code"] for d in old]
    csv_data = make_csv(header, [["r1"] + ["1"] * 41])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"
    assert "40" in resp.json()["detail"]


def test_batch_rejects_more_than_40_new_districts():
    pkg = build(
        [{"code": "O1", "blocks": ["a"]}],
        [{"code": f"N{i}", "blocks": ["a"]} for i in range(41)],
    )
    csv_data = make_csv(["row_id", "O1"], [["r1", "1"]])
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_input"
    assert "40" in resp.json()["detail"]


def test_batch_combination_limit(monkeypatch):
    monkeypatch.setattr(batch, "MAX_REPORTED_COMBINATIONS", 1)
    pkg = simple_package()
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3"],
        [["r1", "1", "2", "3"], ["r2", "", "2", "3"]],
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "invalid_input"
    assert "record 2" in body["detail"]
    assert "combination" in body["detail"]


def test_batch_row_limit(monkeypatch):
    monkeypatch.setattr(batch, "MAX_ROWS", 2)
    pkg = simple_package()
    csv_data = make_csv(
        ["row_id", "O1", "O2", "O3"],
        [["r1", "1", "2", "3"], ["r2", "1", "2", "3"], ["r3", "1", "2", "3"]],
    )
    resp = post_batch(pkg, csv_data)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "invalid_input"
    assert "record 3" in body["detail"]


# --------------------------------------------------------------------------
# The original JSON endpoint is unaffected.
# --------------------------------------------------------------------------
def test_convert_json_endpoint_still_works():
    pkg = simple_package()
    resp = converter.post(
        "/convert",
        json={"package": pkg, "old_populations": {"O1": 10, "O2": 5, "O3": 7}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {c["code"]: c["population"] for c in body["computed"]} == {
        "N1": 15,
        "N2": 7,
    }
