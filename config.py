import os
# --- DEPLOYMENT AYARI ---
# Bu satır, Render'daki kalıcı diskimizin yolunu gösterir.
# Projemiz Render'da çalışırken bu yol kullanılacak.
DATA_DIR = "/data" 

# --- YEREL BİLGİSAYARDA ÇALIŞTIRMAK İÇİN ---
# Eğer kodu gelecekte kendi bilgisayarında denemek istersen, üstteki satırı yorum (#) haline getirip,
# alttaki iki satırın yorumunu kaldırarak projenin mevcut klasörde çalışmasını sağlayabilirsin.
# if not os.path.exists(DATA_DIR):
#     DATA_DIR = os.getcwd()


# --- VERİ TOPLAMA AYARLARI ---
# Sunucuda ilk veritabanı oluşturulurken, geçmiş haberleri çekmeye bu tarihten başla.
ARCHIVE_START_DATE = "2020-01-01" 


# --- DOSYA YOLLARI ---
# Tüm dosya yollarını, yukarıdaki DATA_DIR'a göre otomatik olarak ayarlıyoruz.
KNOWLEDGE_BASE_CSV = os.path.join(DATA_DIR, "knowledge_base.csv")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
# Canlı akıştan gelen haberlerin geçici olarak biriktirileceği dosya
LIVE_BUFFER_CSV = os.path.join(DATA_DIR, "live_buffer.csv")
# Geçmiş verileri çekerken kullanılacak geçici dosya
RAW_NEWS_CSV = os.path.join(DATA_DIR, "temp_raw_news.csv")


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

# --- API ANAHTARLARI İSİMLERİ ---
# .env dosyasındaki anahtar isimleri
GEMINI_API_KEY = "GEMINI_API_KEY"
ALPACA_API_KEY_ENV = "ALPACA_API_KEY"
ALPACA_SECRET_KEY_ENV = "ALPACA_SECRET_KEY"
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"