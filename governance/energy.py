
import json
import time
from contextlib import contextmanager
from typing import Dict, Any, List

import psutil

from config import (
    AVERAGE_WATTS_CPU,
    EMISSION_FACTOR_KG_PER_KWH,
    GOVERNANCE_LOG_FILE,
)
from .audit import log_governance_event


@contextmanager
def energy_tracker(label: str):
    """
    Rough energy tracker:
    - uses process CPU user time delta as proxy
    - estimates Wh and logs as governance event
    """
    proc = psutil.Process()
    cpu_start = proc.cpu_times()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        t1 = time.perf_counter()
        cpu_end = proc.cpu_times()
        elapsed = t1 - t0

        cpu_user_delta = cpu_end.user - cpu_start.user
        watt_seconds = AVERAGE_WATTS_CPU * cpu_user_delta
        watt_hours = watt_seconds / 3600.0
        kg_co2e = (watt_hours / 1000.0) * EMISSION_FACTOR_KG_PER_KWH

        event: Dict[str, Any] = {
            "label": label,
            "elapsed_seconds": float(elapsed),
            "cpu_user_seconds": float(cpu_user_delta),
            "estimated_wh": float(watt_hours),
            "estimated_kg_co2e": float(kg_co2e),
        }
        log_governance_event("ENERGY_USAGE", event)


def load_energy_events() -> List[Dict[str, Any]]:
    path = GOVERNANCE_LOG_FILE
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    events = [json.loads(l) for l in lines]
    return [e for e in events if e.get("event_type") == "ENERGY_USAGE"]


def compute_energy_summary() -> Dict[str, Any]:
    events = load_energy_events()
    if not events:
        return {
            "total_wh": 0.0,
            "total_kwh": 0.0,
            "total_kg_co2e": 0.0,
            "by_label": {},
        }

    by_label: Dict[str, Dict[str, float]] = {}
    total_wh = 0.0
    total_kg = 0.0

    for e in events:
        det = e.get("details", {})
        label = det.get("label", "unknown")
        wh = float(det.get("estimated_wh", 0.0))
        kg = float(det.get("estimated_kg_co2e", 0.0))

        total_wh += wh
        total_kg += kg

        if label not in by_label:
            by_label[label] = {"wh": 0.0, "kg_co2e": 0.0}
        by_label[label]["wh"] += wh
        by_label[label]["kg_co2e"] += kg

    total_kwh = total_wh / 1000.0
    return {
        "total_wh": total_wh,
        "total_kwh": total_kwh,
        "total_kg_co2e": total_kg,
        "by_label": by_label,
    }
