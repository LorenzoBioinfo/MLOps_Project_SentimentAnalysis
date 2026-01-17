from pathlib import Path

# =========================
# PATHS (HOST)
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
HF_CACHE_DIR = PROJECT_ROOT / "huggingface_cache"


# =========================
# PATHS (CONTAINER)
# =========================

APP_DIR = "/app"

CONTAINER_DATA_DIR = f"{APP_DIR}/data"
CONTAINER_MODELS_DIR = f"{APP_DIR}/models"
CONTAINER_REPORTS_DIR = f"{APP_DIR}/reports"
CONTAINER_HF_CACHE_DIR = f"{APP_DIR}/huggingface_cache"


# =========================
# DOCKER IMAGES
# =========================

DATA_PREP_IMAGE = "sentiment-data-prep:latest"
MONITORING_IMAGE = "sentiment-monitoring:latest"
APP_IMAGE = "sentiment-app:latest"


# =========================
# DOCKER VOLUMES
# =========================

COMMON_VOLUMES = {
    str(DATA_DIR): CONTAINER_DATA_DIR,
    str(MODELS_DIR): CONTAINER_MODELS_DIR,
    str(REPORTS_DIR): CONTAINER_REPORTS_DIR,
    str(HF_CACHE_DIR): CONTAINER_HF_CACHE_DIR,
}


# =========================
# ENVIRONMENT VARIABLES
# =========================

COMMON_ENV = {
    "BASE_DIR": APP_DIR,
    "HF_HOME": CONTAINER_HF_CACHE_DIR,
}


# =========================
# APP CONFIG
# =========================

APP_PORT = 8000
APP_HOST_PORT = 5000


# =========================
# PREFECT CONFIG
# =========================

DATA_PREP_TIMEOUT = 1800      
MONITORING_TIMEOUT = 900     
APP_TIMEOUT = None            