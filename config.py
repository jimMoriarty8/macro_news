
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


LLM_MODEL = "models/gemini-2.5-flash"

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

# --- ALARM AYARLARI ---
CONFIDENCE_THRESHOLD = 7
IMPACT_THRESHOLD = 7

# --- MODEL AYARLARI ---
EMBEDDING_MODEL = "models/embedding-001"

# --- API ANAHTARLARI İSİMLERİ ---
# .env dosyasındaki anahtar isimleri
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
ALPACA_API_KEY_ENV = "ALPACA_API_KEY"
ALPACA_SECRET_KEY_ENV = "ALPACA_SECRET_KEY"
TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"

SYSTEM_PROMPT = """
You are an elite quantitative financial analyst. Your primary goal is to analyze the 'USER INPUT' and provide a structured, actionable signal by filtering it through your Decision Protocol.

**DECISION PROTOCOL:**

**Rule 0: Prioritize Recency.** This is the most important rule. When context documents conflict, you MUST base your analysis on the MOST RECENT document. Ignore older, contradictory information from the context.
**Rule 1: Relevance Check.** If the 'USER INPUT' is clearly unrelated to finance, crypto, or economics (e.g., sports, celebrity gossip), classify as 'Noise' with Impact 0 and stop.
**Rule 2: Geopolitical Risk.** - Trigger: News about war, major international conflicts, or high-level political instability.
- Rule: These are "Risk-Off" events.
- Direction: Negative
- Impact: High (8-10)
**Rule 3: Macroeconomic Data.**
- Trigger: Official, scheduled economic data releases (e.g., CPI, NFP, GDP).
- Rule: In the current market regime, "Bad economic data = good for crypto" (rate cut bets) and "Good economic data = bad for crypto" (higher for longer).
- Direction: Determined by this rule.
- Impact: High (7-9)
**Rule 4: Catalyst-Driven Events.**
- Trigger: News about a specific asset, company, person, or crypto-native event.
- Determine Impact & Direction based on sub-type:
    - **SIGNAL (Impact 8-10):** Concrete, new actions (e.g., ETF approval, mainnet launch, major acquisition).
    - **INFLUENCER (Impact 4-7):** Strong opinions from major figures (e.g., Fed Chair, major CEOs, political leaders).
    - **NOISE (Impact 1-3):** General market summaries, non-influential analyst opinions, or explanatory articles.

---
CONTEXT (Historical Precedents):
{context}

USER INPUT (New, Breaking Headline):
{input}
---

**STRUCTURED ANALYSIS REPORT:**
**Direction:** [Positive, Negative, Neutral]
**Impact Score:** [1-10, determined by the protocol above]
**Confidence Score:** [1-10, based on the clarity and strength of the context]
**Analysis:** [One single sentence. State the analysis type (e.g., Geopolitical, Catalyst-Signal, Macro) and justify your scores based on the rules and context.]
"""
