import os

# --- DEPLOYMENT AYARI (ŞİMDİLİK YORUMDA KALSIN) ---
# Render.com'a yüklediğimizde bu DATA_DIR'ı kullanacağız.
# Yerelde çalışırken bu satır yorumda kalmalı.
# DATA_DIR = "/data" 

# --- YEREL ÇALIŞMA AYARI ---
# Eğer yukarıdaki DATA_DIR yorumdaysa, mevcut klasörü baz al.
DATA_DIR = os.getcwd()

ARCHIVE_START_DATE = "2020-01-01" 

# --- DOSYA YOLLARI ---
# Tüm dosya yollarını bu merkezi yerden yöneteceğiz.
LIVE_BUFFER_CSV = os.path.join(DATA_DIR, "live_buffer.csv")
RAW_NEWS_CSV = os.path.join(DATA_DIR, "temp_raw_news.csv")
KNOWLEDGE_BASE_CSV = os.path.join(DATA_DIR, "knowledge_base.csv")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# --- SEMBOL AYARLARI ---
# Takip edilecek semboller
SYMBOLS_TO_TRACK = [
    'BTC/USD', "BTC", 'BTCUSD',
    'ETH/USD', "ETH", 'ETHUSD',
    "SOL/USD", "SOL", 'SOLUSD',
    "XRP/USD", "XRP", 'XRPUSD',
    "BNB/USD", "BNB", 'BNBUSD',
    'SPY', 'QQQ'
]

# --- MODEL AYARLARI ---
EMBEDDING_MODEL = "models/embedding-001"