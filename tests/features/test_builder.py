import math

import numpy as np
import pandas as pd
import pytest

from argus.core.config import FeaturesConfig
from argus.features import PandasFeatureBuilder
from argus.features.entropy import count_unique, shannon_entropy
from argus.features.frame import records_to_frame

from .helpers import rec


def _build(records: list, **cfg: object) -> pd.DataFrame:
    config = FeaturesConfig(entities=["source_ip"], windows=["1m", "5m"], **cfg)
    return PandasFeatureBuilder(config).build(records).frame


def _last(frame: pd.DataFrame) -> pd.Series:
    return frame.iloc[-1]


def test_entropy_helpers() -> None:
    assert shannon_entropy(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(2.0)
    assert shannon_entropy(np.array([5.0, 5.0])) == 0.0
    assert str(shannon_entropy(np.array([5.0]))) == "0.0"
    assert shannon_entropy(np.array([-1.0, -1.0])) == 0.0
    assert count_unique(np.array([1.0, 1.0, 2.0, -1.0])) == 2.0


def test_frame_is_stably_sorted() -> None:
    frame = records_to_frame([rec(5), rec(1), rec(1, ip="2.2.2.2")])
    assert frame["row_id"].tolist() == [1, 2, 0]


def test_basic_counts_and_errors() -> None:
    recs = [
        rec(0, path="/a"),
        rec(10, path="/a", status=404),
        rec(20, path="/b", status=500),
        rec(30, path="/login", status=401),
        rec(40, path="/admin", status=403),
    ]
    last = _last(_build(recs))
    assert last["request_count_1m"] == 5
    assert last["error_count_1m"] == 4
    assert last["client_error_count_1m"] == 3
    assert last["server_error_count_1m"] == 1
    assert last["error_rate_1m"] == pytest.approx(0.8)
    assert last["failed_auth_count_1m"] == 2
    assert last["unique_paths_1m"] == 4
    assert last["distinct_status_codes_1m"] == 5
    assert last["unique_user_agents_1m"] == 1


def test_window_boundary_is_exclusive_on_the_left() -> None:
    last = _last(_build([rec(0), rec(60), rec(120)]))
    assert last["request_count_1m"] == 1
    assert last["request_count_5m"] == 3


def test_entropy_and_new_path_ratio() -> None:
    recs = [rec(0, path="/a"), rec(1, path="/b"), rec(2, path="/a"), rec(3, path="/b")]
    last = _last(_build(recs))
    assert last["path_entropy_1m"] == pytest.approx(1.0)
    assert last["new_path_ratio_1m"] == pytest.approx(0.5)


def test_new_path_is_relative_to_entity_history_not_window() -> None:
    recs = [rec(0, path="/a"), rec(600, path="/a")]
    last = _last(_build(recs))
    assert last["new_path_ratio_1m"] == 0.0
    assert last["new_path_ratio_5m"] == 0.0


def test_single_event_and_null_fields() -> None:
    last = _last(_build([rec(0, path=None, status=None, nbytes=None, ua=None)]))
    assert last["bytes_sent_std_1m"] == 0.0
    assert last["bytes_sent_mean_1m"] == 0.0
    assert last["error_rate_1m"] == 0.0
    assert last["unique_paths_1m"] == 0
    assert last["path_entropy_1m"] == 0.0
    assert last["new_path_ratio_1m"] == 0.0
    assert last["request_rate_zscore_1m"] == 0.0
    assert not any(math.isnan(v) for v in last[_build([rec(0)]).columns[4:]].tolist())


def test_bytes_mean_std_skip_nulls() -> None:
    last = _last(_build([rec(0, nbytes=100), rec(1, nbytes=None), rec(2, nbytes=300)]))
    assert last["bytes_sent_mean_1m"] == pytest.approx(200.0)
    assert last["bytes_sent_std_1m"] == pytest.approx(141.421356, abs=1e-5)


def test_null_status_excluded_from_error_rate_denominator() -> None:
    last = _last(_build([rec(0, status=500), rec(1, status=None), rec(2, status=200)]))
    assert last["error_rate_1m"] == pytest.approx(0.5)


def test_temporal_features() -> None:
    frame = _build([rec(0), rec(90)])
    first, second = frame.iloc[0], frame.iloc[1]
    assert first["hour_of_day"] == 10 and first["day_of_week"] == 2
    assert first["is_weekend"] == 0 and first["is_business_hours"] == 1
    assert first["seconds_since_last_event_same_entity"] == -1.0
    assert first["seconds_since_last_event_same_entity_is_first"] == 1
    assert second["seconds_since_last_event_same_entity"] == 90.0
    assert second["seconds_since_last_event_globally_is_first"] == 0


def test_dedup_one_row_per_entity_minute() -> None:
    frame = _build([rec(s) for s in range(0, 50, 5)] + [rec(65)])
    assert len(frame) == 2
    assert frame["request_count_1m"].tolist() == [10, 9]


def test_business_hours_config() -> None:
    frame = _build([rec(0)], business_hours_start=11, business_hours_end=17)
    assert frame.iloc[0]["is_business_hours"] == 0


def test_burst_after_constant_history_has_high_zscore() -> None:
    steady = [rec(i * 120.0) for i in range(8)]
    burst = [rec(1000 + i * 0.5) for i in range(40)]
    frame = _build(steady + burst)
    assert frame["request_rate_zscore_1m"].max() > 5
    assert frame["request_rate_zscore_1m"].iloc[0] == 0.0


def test_null_entities_are_skipped_with_finding() -> None:
    config = FeaturesConfig(entities=["source_ip", "user"], windows=["1m"])
    matrix = PandasFeatureBuilder(config).build([rec(0, user="bob"), rec(1, user=None)])
    assert set(matrix.frame["entity_type"]) == {"source_ip", "user"}
    assert [f.code for f in matrix.findings] == ["entity_null_rows"]
    assert (matrix.frame["entity_type"] == "user").sum() == 1


def test_combined_entity_key() -> None:
    config = FeaturesConfig(entities=["source_ip+user"], windows=["1m"])
    frame = PandasFeatureBuilder(config).build([rec(0, user="bob")]).frame
    assert frame["entity"].tolist() == ["1.1.1.1|bob"]


def test_empty_input() -> None:
    matrix = PandasFeatureBuilder().build([])
    assert matrix.frame.empty and [f.code for f in matrix.findings] == ["no_records"]


def test_window_columns() -> None:
    matrix = PandasFeatureBuilder(FeaturesConfig(windows=["1m", "5m"])).build([rec(0)])
    cols = matrix.window_columns("5m")
    assert "request_count_5m" in cols and "request_count_1m" not in cols
    assert "hour_of_day" in cols
    with pytest.raises(ValueError, match="unknown window"):
        matrix.window_columns("1h")


def test_deterministic_byte_identical() -> None:
    recs = [
        rec(
            i * 3.7,
            ip=f"10.0.0.{i % 4}",
            user=("u1" if i % 3 else None),
            path=f"/p{i % 5}",
            status=[200, 401, 404, 500][i % 4],
            nbytes=i * 10 if i % 7 else None,
        )
        for i in range(120)
    ]
    first = PandasFeatureBuilder().build(recs).to_csv()
    second = PandasFeatureBuilder().build(list(recs)).to_csv()
    assert first == second and len(first) > 1000
