import os


IMAGE_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DATA_DIR = os.environ.get("APP_DATA_DIR", BASE_DIR)

X_PATH = "model/X_224.npy"
Y_PATH = "model/Y_224.npy"
DATA_PATH = "model/data_224.npy"
EFFICIENT_WEIGHTS_PATH = "model/efficient_weights_224.hdf5"
LEGACY_EFFICIENT_WEIGHTS_PATH = "model/efficient_weights.hdf5"
VGG_WEIGHTS_PATH = "model/vgg_weights_224.hdf5"
RESNET_WEIGHTS_PATH = "model/resnet_weights_224.hdf5"
METRIC_PATH = "model/metric_224.npy"
CM_PATH = "model/cm_224.npy"

LEGACY_METRIC_PATH = "model/metric.npy"
LEGACY_CM_PATH = "model/cm.npy"

DATASET_DIR = "Dataset"
STATIC_DIR = os.path.join("DiabeticApp", "static")
SQLITE_DB_PATH = os.path.join(APP_DATA_DIR, "app.sqlite3")

MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "root"),
    "password": os.environ.get("MYSQL_PASSWORD", "root"),
    "database": os.environ.get("MYSQL_DATABASE", "diabetic"),
    "charset": "utf8",
}

POSTGRES_CONFIG = {
    "host": os.environ.get("AZURE_POSTGRESQL_HOST", os.environ.get("DB_HOST", "127.0.0.1")),
    "port": int(os.environ.get("AZURE_POSTGRESQL_PORT", os.environ.get("DB_PORT", "5432"))),
    "user": os.environ.get("AZURE_POSTGRESQL_USER", os.environ.get("DB_USER", "")),
    "password": os.environ.get("AZURE_POSTGRESQL_PASSWORD", os.environ.get("DB_PASSWORD", "")),
    "dbname": os.environ.get("AZURE_POSTGRESQL_NAME", os.environ.get("DB_NAME", "")),
    "sslmode": os.environ.get("AZURE_POSTGRESQL_SSLMODE", os.environ.get("DB_SSLMODE", "require")),
}

APP_DB_BACKEND = os.environ.get("APP_DB_BACKEND", os.environ.get("DB_ENGINE", "sqlite")).lower()
