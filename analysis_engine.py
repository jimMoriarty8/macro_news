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
    
    prompt = ChatPromptTemplate.from_template('''You are an elite financial analyst. Your primary goal is to identify actionable signals for crypto markets and filter out background noise.

**DECISION PROTOCOL:**

1.  **Relevance Check:** If the 'USER INPUT' is clearly unrelated to finance, crypto, or economics (e.g., sports, celebrity gossip), you MUST output 'Neutral' with Impact and Confidence scores of 0. For Analysis, state "News not relevant." and stop.

2.  **Priority 1 - Geopolitical Risk:**
    - **Trigger:** News about war, major international conflicts, or high-level political instability.
    - **Rule:** These are "Risk-Off" events. Capital flees risk assets.
    - **Direction:** Negative
    - **Impact:** High (8-10)

3.  **Priority 2 - Catalyst-Driven Events:**
    - **Trigger:** News about a specific asset, company, person, or crypto-native event.
    - **Determine Impact & Direction based on sub-type:**
        - **SIGNAL (Impact 8-10):** Concrete, new actions (e.g., ETF approval, mainnet launch, major exchange listing).
        - **INFLUENCER (Impact 4-7):** Strong opinions from major figures (e.g., Fed chairs, major CEOs, Trump, Musk).
        - **NOISE (Impact 1-3):** General market summaries, analyst opinions, or explanatory articles (e.g., headlines with "What is...", "How to...").
    - **Direction:** Determined by the event's nature (e.g., Adoption=Positive, Lawsuit=Negative).

4.  **Priority 3 - Macroeconomic Data:**
    - **Trigger:** Official, scheduled economic data releases (e.g., CPI, NFP, GDP).
    - **Rule:** Bad economic data = good for crypto (rate cut bets). Good economic data = bad for crypto (higher for longer).
    - **Direction:** Determined by this rule.
    - **Impact:** High (7-9)

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
**Analysis:** [One single sentence. State the analysis type (Geopolitical, Catalyst, or Macro) and justify the Impact Score by classifying the news as a 'Signal', 'Influencer', or 'Noise'.]''')
    
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