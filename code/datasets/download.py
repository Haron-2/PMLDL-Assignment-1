from urllib.request import urlretrieve
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)

urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', RAW_DIR / 'winequality-red.csv')
urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', RAW_DIR / 'winequality-white.csv')