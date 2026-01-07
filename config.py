
from pathlib import Path

# ==================== System Metadata ====================
SYSTEM_NAME = "Governed Agri Stress Risk Advisor"
SYSTEM_VERSION = "1.0.0"
PROVIDER = "Jit Research Lab"
RISK_LEVEL = "high-risk-candidate"

# Protected attributes for fairness diagnostics
PROTECTED_ATTRIBUTES = ["gender", "farm_size_class"]

# Child threshold for governance
CHILD_AGE_THRESHOLD = 16

# ==================== Paths ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
MODEL_CARD_DIR = BASE_DIR / "model_cards"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MODEL_CARD_DIR.mkdir(exist_ok=True)

INFERENCE_LOG_FILE = LOG_DIR / "inference_log.jsonl"
GOVERNANCE_LOG_FILE = LOG_DIR / "governance_events.jsonl"
REGISTRY_FILE = MODEL_CARD_DIR / "algorithm_registry.json"
STRESS_MODEL_CARD_FILE = MODEL_CARD_DIR / "stress_risk_model_card.json"

# ==================== Policy Modes ====================
POLICY_MODES = ["Research", "Public Service"]
DEFAULT_POLICY_MODE = "Research"

PUBLIC_SERVICE_RULES = {
    "require_consent": True,
    "block_child_inference": True,
    "allow_unlicensed_data": False,
}

# ==================== Energy & Carbon ====================
# Approximate CPU power (Watts) for demo purposes
AVERAGE_WATTS_CPU = 35.0
# Rough EU grid emission factor (kg CO2e per kWh)
EMISSION_FACTOR_KG_PER_KWH = 0.233
