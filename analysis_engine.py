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
    
    prompt = ChatPromptTemplate.from_template('''You are a world-class financial analyst, specializing in both macroeconomic trends and specific crypto catalysts. Your task is to analyze the 'USER INPUT' (a new headline) by first categorizing it and then applying the appropriate analytical framework based on the provided 'CONTEXT'.

---
**ANALYTICAL FRAMEWORKS**

**FRAMEWORK 1: Macroeconomic Regime (FED-Centric)**
(Use this framework EXCLUSIVELY for official economic data releases from government bodies like the Bureau of Labor Statistics, Central Banks, etc. Examples: CPI, PPI, Non-Farm Payrolls (NFP), GDP, Retail Sales.)
- **Rule 1.1 (Bad News is Good News):** Data indicating a weakening economy (e.g., lower-than-expected inflation, rising unemployment) is considered **POSITIVE** for crypto, as it increases the probability of Fed rate cuts (more liquidity).
- **Rule 1.2 (Good News is Bad News):** Data indicating a strong economy (e.g., high retail sales, strong job reports) is considered **NEGATIVE** for crypto, as it reinforces a "higher for longer" interest rate policy (less liquidity).

**FRAMEWORK 2: Catalyst-Driven Regime (Specific Events)**
(Use this framework for news pertaining to a specific coin, company, person, or event.)
- **Rule 2.1 (Hype & Sentiment):** A strong supportive statement about crypto from a globally recognized, market-moving figure (e.g., Elon Musk, Donald Trump) is a very strong **POSITIVE** signal with high immediate impact potential.
- **Rule 2.2 (Adoption & Technology):** An announcement of a major corporation (e.g., Google, BlackRock) using a coin, a major exchange listing (Coinbase, Binance), or the successful completion of a significant technological upgrade is considered **POSITIVE** for the relevant asset and general market sentiment.
- **Rule 2.3 (Regulation):** News such as a country banning crypto or the SEC filing a lawsuit against a major project is considered **NEGATIVE**. Favorable regulatory news is **POSITIVE**.
- **Rule 2.4 (Indirect Impact / Sector Sentiment):** Positive or negative news about major tech companies (like MSFT, NVDA) or traditional finance does not have a direct impact on crypto. Interpret such news as an indirect **sentiment signal** on the tech/risk-appetite sector and keep its 'Impact Score' low (in the 1-3 range).

---
**YOUR TASK (Step-by-Step):**

1.  **Categorize (MOST IMPORTANT STEP):** First, analyze the 'USER INPUT'. **The Tie-Breaker Rule: If the news headline mentions a specific company, asset, or person by name (e.g., 'Bitcoin', 'Microsoft', 'Trump'), you MUST default to using Framework 2 (Catalyst-Driven).** You should only use Framework 1 if the news is a broad, official economic data release without a specific company focus.
2.  **Analyze:** Apply the rules from the relevant framework chosen in Step 1.
3.  **Score:** Assign Impact and Confidence scores. A concrete action (e.g., "Coinbase lists coin X") should have a higher 'Confidence Score' than a mere opinion ("Trump likes Bitcoin"). The power of the news source (e.g., Trump vs. an unknown analyst) affects the 'Impact Score'.
4.  **Report:** Fill out the structured report below. In the 'Analysis' section, you MUST explicitly state which framework you used and why.

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
**Analysis:** [A concise, 1-2 sentence reasoning. Explicitly state WHICH framework (Macroeconomic or Catalyst-Driven) you used and why the news fits its rules.]''')
    
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