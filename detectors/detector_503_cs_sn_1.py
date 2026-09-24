from __future__ import annotations

from detectors.detector_506_cs_sn_1 import Detector506CsSn1


class Detector503CsSn1(Detector506CsSn1):
    """503 identity for the shared fixed-threshold polygon detector contract."""

    detector_id = "503-CS-SN-1"
    detector_name = "global_polygon_detector_503"
    display_name = "503-CS-SN-1 global polygon detector"
    defect_type = "503_cs_sn_1_polygon_ng"
    preprocess_plan_name = "503_cs_sn_1_preprocess"
    # 503 shares the polygon schema with 506, but owns an independent declaration so
    # future recipe changes for either registered Detector cannot alias class state.
    default_params = {**Detector506CsSn1.default_params}
    PARAM_SPEC = {**Detector506CsSn1.PARAM_SPEC}
