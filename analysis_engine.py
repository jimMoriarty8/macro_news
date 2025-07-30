# analysis_engine.py


import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config
import re
import requests
from binance.client import Client as BinanceClient

def initialize_analyst_assistant():
    """
    Vektör veritabanını, LLM'i ve temel RAG parçalarını kurar.
    main.py'nin hem analiz hem de veritabanı güncellemesi yapabilmesi için
    gerekli olan 3 aracı (retriever, document_chain, vector_store) döndürür.
    """
    print("RAG Analist Asistanı başlatılıyor...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı! Lütfen .env dosyasını veya Render ayarlarını kontrol edin.")

    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2, google_api_key=api_key)
    
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Mevcut vektör veritabanı '{config.CHROMA_DB_PATH}' klasöründen yükleniyor...")
        vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
        print("Veritabanı başarıyla yüklendi.")
    else:
        print(f"UYARI: Henüz bir veritabanı bulunamadı. '{config.KNOWLEDGE_BASE_CSV}' dosyasından oluşturulacak...")
        try:
            df = pd.read_csv(config.KNOWLEDGE_BASE_CSV)
            documents = [
                Document(
                    page_content=row['rag_content'],
                    metadata={'source': row['source'], 'title': row['headline'], 'publish_date': row['timestamp']}
                ) for index, row in df.iterrows()
            ]
            vector_store = Chroma.from_documents(
                documents=documents, 
                embedding=embeddings, 
                persist_directory=config.CHROMA_DB_PATH
            )
            print(f"Yeni veritabanı '{config.CHROMA_DB_PATH}' klasöründe başarıyla oluşturuldu.")
        except FileNotFoundError:
             print(f"HATA: '{config.KNOWLEDGE_BASE_CSV}' dosyası bulunamadı. Lütfen önce veri toplama ve işleme script'lerini çalıştırın.")
             exit()


    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    
    # --- DÜZELTME BURADA ---
    # Önce prompt metnini bir değişkene atıyoruz.
    system_prompt = """
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
    # Sonra bu değişkeni fonksiyona parametre olarak veriyoruz.
    prompt = ChatPromptTemplate.from_template(system_prompt)
    # --- DÜZELTME SONU ---
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    print("\nAnalist Asistanı'nın tüm parçaları hazır.")
    print("="*50)
    
    return retriever, document_chain, vector_store

# --- Diğer Yardımcı Fonksiyonlar ---
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
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    
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
    except (AttributeError, IndexError, ValueError):
        print(f"AYRIŞTIRMA HATASI: Rapor beklenen formatta değil.")
        return None
