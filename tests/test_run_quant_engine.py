import json

import numpy as np

from scripts.run_quant_engine import save_json_result


def test_save_json_result_serializes_numpy_scalars_and_writes_latest_report(tmp_path):
    payload = {"market_open": np.bool_(True), "score": np.int64(72)}

    report_path = save_json_result(payload, str(tmp_path))

    assert report_path is not None
    assert json.loads((tmp_path / "latest_report.json").read_text()) == {
        "market_open": True,
        "score": 72,
    }
