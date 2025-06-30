# analysis_engine.py

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config

def initialize_analyst_assistant():
    """
    Vektör veritabanını, LLM'i ve temel RAG parçalarını kurar.
    main.py'nin hem analiz hem de veritabanı güncellemesi yapabilmesi için
    gerekli olan 3 aracı (retriever, document_chain, vector_store) döndürür.
    """
    print("RAG Analist Asistanı başlatılıyor...")
    # API anahtarını doğrudan ortam değişkenlerinden al
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Hata mesajını aranan anahtarla tutarlı hale getiriyoruz.
        raise ValueError("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı! Lütfen Render'daki Environment Group'u kontrol edin.")

    # DÜZELTME 2: Anahtarı DOĞRUDAN modellere parametre olarak iletiyoruz.
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    
    # Veritabanı var mı diye kontrol et, yoksa boş oluştur.
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Mevcut vektör veritabanı '{config.CHROMA_DB_PATH}' klasöründen yükleniyor...")
        vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
        print("Veritabanı başarıyla yüklendi.")
    else:
        print("UYARI: Henüz bir veritabanı bulunamadı.")
        print("Sıfırdan BOŞ bir vektör veritabanı oluşturuluyor...")
        # Kütüphanenin hata vermemesi için içeriği boş olan geçici bir dökümanla başlatıyoruz
        placeholder_doc = Document(page_content="initialization_document")
        vector_store = Chroma.from_documents(
            documents=[placeholder_doc], 
            embedding=embeddings, 
            persist_directory=config.CHROMA_DB_PATH
        )
        print(f"Boş veritabanı '{config.CHROMA_DB_PATH}' klasöründe başarıyla oluşturuldu.")

    # LangChain araçlarını oluştur
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    
    prompt = ChatPromptTemplate.from_template('''You are a world-class financial analyst and trading assistant. Your primary goal is to identify actionable, market-moving events ('Signals') and filter out speculative commentary or summaries of existing sentiment ('Noise'). Analyze the 'USER INPUT' by categorizing it, evaluating its actionability, and then applying the appropriate analytical framework.

---
**ANALYTICAL FRAMEWORKS**

**FRAMEWORK 1: Macroeconomic Regime (FED-Centric)**
(Use this framework EXCLUSIVELY for new, scheduled, official economic data releases. Examples: CPI, NFP, GDP data.)
- **Rule 1.1 (Bad News is Good News):** Weak economic data is **POSITIVE** for crypto (signals potential rate cuts).
- **Rule 1.2 (Good News is Bad News):** Strong economic data is **NEGATIVE** for crypto (signals "higher for longer" rates).

**FRAMEWORK 2: Catalyst-Driven Regime (Specific Events)**
(Use this for news about specific assets, companies, people, or regulations.)
- **Rule 2.1 (Hype & Sentiment):** A direct, strong statement from a globally recognized market-mover (e.g., Musk, Trump) is a **POSITIVE** signal.
- **Rule 2.2 (Adoption & Technology):** A major exchange listing, corporate adoption, or tech upgrade is **POSITIVE**.
- **Rule 2.3 (Regulation):** A direct regulatory action (e.g., ETF approval, lawsuit) is a high-impact event.

---
**SCORING GUIDELINES (ACTIONABILITY ASSESSMENT)**

This is the most critical part of your analysis. You must rate the 'Impact Score' based on whether the news is a true 'Signal' or just 'Noise'.

- **High Impact (8-10): The SIGNAL Zone.** Reserved for **new, unexpected, and concrete data or events.**
  - *Examples:* A surprise CPI number, an official SEC decision on an ETF, a major exchange suddenly listing a new asset, a hack.
  - *Your thought process:* "This is new information. The market did not know this 5 minutes ago. This forces an immediate re-evaluation of prices."

- **Medium Impact (4-7): The INFLUENCER Zone.** Reserved for **statements of intent, strong opinions from powerful figures, or expected but significant events.**
  - *Examples:* A Fed governor reiterating a known stance, Trump's positive comments on crypto, a project announcing a future mainnet date.
  - *Your thought process:* "This is not a concrete action yet, but it strongly influences future expectations."

- **Low Impact (1-3): The NOISE Zone.** Reserved for **summaries of past events, general market commentary, or analyst opinions.**
  - *Examples:* "Markets rise on rate cut bets" (summary of sentiment), "Analyst says stock X could rise" (opinion), "Bitcoin is up 5% this week" (summary of past price action).
  - *Your thought process:* "This headline is not providing new, actionable information. It's describing what has already happened or what everyone is already talking about. This is not a trigger for an immediate trade."

---
**YOUR TASK (Step-by-Step):**

1.  **Categorize:** First, decide if the headline fits Framework 1 (official data) or Framework 2 (specific event). **Tie-Breaker Rule: If a specific company/person is named, always default to Framework 2.**
2.  **Evaluate Actionability:** Using the **SCORING GUIDELINES**, determine if the news is a 'Signal' or 'Noise' and decide on a preliminary 'Impact Score' range.
3.  **Analyze & Report:** Apply the rules from the chosen framework and fill out the report below. Your final 'Impact Score' must be consistent with your actionability evaluation. In the 'Analysis' section, explicitly state your reasoning for the Impact Score, referencing the 'Signal' vs. 'Noise' concept.

---
CONTEXT (Historical Precedents):
{context}

USER INPUT (New, Breaking Headline):
{input}
---

**STRUCTURED ANALYSIS REPORT:**
**Direction:** [Positive, Negative, Neutral]
**Impact Score:** [1-10]
**Confidence Score:** [1-10]
**Analysis:** [A concise reasoning. State the framework used. Crucially, justify the 'Impact Score' by explaining if the news is an actionable 'Signal' or background 'Noise' based on the guidelines.]''')
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    print("\nAnalist Asistanı'nın tüm parçaları hazır.")
    print("="*50)
    
    # main.py'nin ihtiyaç duyduğu 3 temel aracı döndür
    return retriever, document_chain, vector_store

# --- Diğer Yardımcı Fonksiyonlar ---
# Projenin diğer dosyalarında (main.py gibi) kullanılacak olan
# parse_analyst_report, send_telegram_message ve get_btc_price
# fonksiyonlarını da bu "motor" dosyasında merkezi olarak tutmak iyi bir fikirdir.

import re
import requests
from binance.client import Client as BinanceClient

def get_btc_price():
    """Anlık BTC/USDT fiyatını çeker."""
    try:
        client = BinanceClient()
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        price = float(ticker['price'])
        return f"${price:,.2f}"
    except Exception as e:
        print(f"UYARI: Anlık BTC fiyatı çekilemedi. Hata: {e}")
        return "Price N/A"

def send_telegram_message(message):
    """Belirtilen mesajı Telegram'a gönderir."""
    TOKEN = os.getenv(config.TELEGRAM_BOT_TOKEN_ENV)
    CHAT_ID = os.getenv(config.TELEGRAM_CHAT_ID_ENV)
    
    if not TOKEN or not CHAT_ID:
        print("\n>> UYARI: Telegram bilgileri eksik. Mesaj gönderilemedi.")
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("\n>>> ALARM TELEGRAM'A BAŞARIYLA GÖNDERİLDİ <<<")
        else:
            print(f"\n>> HATA: Telegram'a mesaj gönderilemedi. Durum Kodu: {response.status_code}, Cevap: {response.text}")
    except Exception as e:
        print(f"\n>> HATA: Telegram mesajı gönderilirken bir sorun oluştu: {e}")

def parse_analyst_report(report_text):
    """LLM'den gelen metin raporunu ayrıştırır."""
    try:
        direction = re.search(r"\*\*?Direction:\*\*?\s*\[?([^\]\n]+)\]?", report_text, re.IGNORECASE).group(1).strip()
        impact = int(re.search(r"\*\*?Impact Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        confidence = int(re.search(r"\*\*?Confidence Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        analysis_match = re.search(r"\*\*?Analysis:\*\*?\s*([\s\S]*)", report_text, re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else "Analiz metni bulunamadı."
        return {"direction": direction, "impact": impact, "confidence": confidence, "analysis": analysis}
    except (AttributeError, IndexError, ValueError) as e:
        print(f"AYRIŞTIRMA HATASI: Rapor beklenen formatta değil. Hata: {e}")
        return None