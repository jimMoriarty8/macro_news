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
    
    
    # Gerekli araçları ve modelleri başlat
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
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
    
    prompt = ChatPromptTemplate.from_template("""
    You are a world-class quantitative financial analyst. Your task is to analyze the 'USER INPUT' (a new headline) based on the provided 'CONTEXT' (historical news) and the CURRENT MARKET REGIME.

    **CURRENT MARKET REGIME (Primary Rule Set):**
    You are operating in a market environment where the central bank's (FED) monetary policy is the primary driver of risk asset prices (like crypto). Therefore, you MUST adhere to the following logic:
    - Rule 1 (Bad News is Good News): Data indicating a *cooling* economy (e.g., lower-than-expected inflation, rising unemployment) is considered **POSITIVE** for crypto, as it increases the probability of Fed rate cuts (more liquidity).
    - Rule 2 (Good News is Bad News): Data indicating a *stronger-than-expected* economy (e.g., high durable goods orders, strong jobs reports) is considered **NEGATIVE** for crypto, as it reinforces a "higher for longer" interest rate policy (less liquidity).
    - Rule 3 (Source Authority): Always weigh your analysis by the source's authority as a secondary factor. A formal FOMC decision is more important than a regional governor's comments.

    ---
    CONTEXT (Historical Precedents):
    {context}

    USER INPUT (New, Breaking Headline):
    {input}
    ---

    **STRUCTURED ANALYSIS REPORT (filtered through the market regime rules):**
    **Direction:** [Positive, Negative, Neutral]
    **Impact Score:** [1-10]
    **Confidence Score:** [1-10]
    **Analysis:** [A concise, 1-2 sentence reasoning. Explicitly state HOW the news fits into the "Good News is Bad News" or "Bad News is Good News" regime.]
    """)
    
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