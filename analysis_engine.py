# app.py (MODİFİYE EDİLMİŞ HALİ)

import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import requests
from dotenv import load_dotenv
from binance.client import Client as BinanceClient
import config

# .env dosyasındaki tüm değişkenleri yükle
load_dotenv()

# --- 1. AYARLAR ---
# Bu ayarlar artık main_controller.py tarafından kullanılacak ama burada kalabilir.
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
KNOWLEDGE_BASE_CSV = "knowledge_base.csv"
CHROMA_DB_PATH = "./chroma_db"

# --- 2. YARDIMCI FONKSİYONLAR (DEĞİŞİKLİK YOK) ---
def get_btc_price():
    """python-binance kütüphanesini kullanarak anlık BTC/USDT fiyatını çeker."""
    try:
        client = BinanceClient()
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        price = float(ticker['price'])
        return f"${price:,.2f}"
    except Exception as e:
        print(f"UYARI: Anlık BTC fiyatı (python-binance) ile çekilemedi. Hata: {e}")
        return "Price N/A"

def send_telegram_message(message):
    """Belirtilen mesajı Telegram'a gönderir."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n>> UYARI: .env dosyasında Telegram bilgileri eksik. Mesaj gönderilemedi.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("\n>>> ALARM TELEGRAM'A BAŞARIYLA GÖNDERİLDİ <<<")
        else:
            print(f"\n>> HATA: Telegram'a mesaj gönderilemedi. Durum Kodu: {response.status_code}, Cevap: {response.text}")
    except Exception as e:
        print(f"\n>> HATA: Telegram mesajı gönderilirken bir sorun oluştu: {e}")

def parse_analyst_report(report_text):
    """LLM'den gelen metin raporunu, Markdown formatı gibi farklılıklara karşı dayanıklı bir şekilde ayrıştırır."""
    try:
        direction = re.search(r"\*\*?Direction:\*\*?\s*\[?([^\]\n]+)\]?", report_text, re.IGNORECASE).group(1).strip()
        impact = int(re.search(r"\*\*?Impact Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        confidence = int(re.search(r"\*\*?Confidence Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        analysis_match = re.search(r"\*\*?Analysis:\*\*?\s*([\s\S]*)", report_text, re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else "Analiz metni bulunamadı."
        return {"direction": direction, "impact": impact, "confidence": confidence, "analysis": analysis}
    except (AttributeError, IndexError, ValueError) as e:
        print(f"AYRIŞTIRMA HATASI: Rapor beklenen formatta değil. Hata: {e}")
        print(f"Sorunlu Rapor Metni:\n---\n{report_text}\n---")
        return None


# --- 3. RAG SİSTEMİ KURULUM FONKSİYONU ---
def initialize_analyst_assistant():
    """
    Vektör veritabanını, LLM'i ve RAG zincirini kurar.
    YENİ: Veritabanı veya CSV dosyası olmadığında sıfırdan boş bir DB oluşturabilir.
    """
    print("RAG Analist Asistanı başlatılıyor...")
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    
    # Chroma veritabanı var mı diye kontrol et
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Mevcut vektör veritabanı '{config.CHROMA_DB_PATH}' klasöründen yükleniyor...")
        vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
        print("Veritabanı başarıyla yüklendi.")
    else:
        # Yoksa, knowledge_base.csv var mı diye bak
        if os.path.exists(config.KNOWLEDGE_BASE_CSV):
            print(f"'{config.KNOWLEDGE_BASE_CSV}' dosyasından yeni bir vektör veritabanı oluşturuluyor...")
            df = pd.read_csv(config.KNOWLEDGE_BASE_CSV)
            documents = [
                Document(
                    page_content=row['rag_content'],
                    metadata={'source': row.get('source'), 'title': row.get('title'), 'publish_date': row.get('timestamp')}
                ) for index, row in df.iterrows()
            ]
            vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=config.CHROMA_DB_PATH)
            print(f"Yeni veritabanı oluşturuldu ve '{config.CHROMA_DB_PATH}' klasörüne kaydedildi.")
        else:
            # HİÇBİR ŞEY YOKSA (YENİ KURULUM): Boş bir veritabanı oluştur.
            print("UYARI: Ne mevcut bir vektör DB ne de knowledge_base.csv bulundu.")
            print("Sıfırdan BOŞ bir vektör veritabanı oluşturuluyor. Veritabanını doldurmak için güncelleyiciyi çalıştırın.")
            # Boş bir document listesi ile Chroma'yı başlat
            vector_store = Chroma.from_documents(documents=[], embedding=embeddings, persist_directory=config.CHROMA_DB_PATH)
            print(f"Boş veritabanı başarıyla oluşturuldu ve '{config.CHROMA_DB_PATH}' klasörüne kaydedildi.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template("""
    You are a world-class quantitative financial analyst. Your task is to analyze the 'USER INPUT' (a new headline) based on the provided 'CONTEXT' (historical news) and the CURRENT MARKET REGIME.

    **CURRENT MARKET REGIME (Primary Rule Set):**
    You are operating in a market environment where the central bank's (FED) monetary policy is the primary driver of risk asset prices (like crypto). Therefore, you MUST adhere to the following logic:
    - **Rule 1 (Bad News is Good News):** Data indicating a *cooling* economy (e.g., lower-than-expected inflation, rising unemployment) is considered **POSITIVE** for crypto, as it increases the probability of Fed rate cuts (more liquidity).
    - **Rule 2 (Good News is Bad News):** Data indicating a *stronger-than-expected* economy (e.g., high durable goods orders, strong jobs reports) is considered **NEGATIVE** for crypto, as it reinforces a "higher for longer" interest rate policy (less liquidity).
    - **Rule 3 (Source Authority):** Always weigh your analysis by the source's authority as a secondary factor. A formal FOMC decision is more important than a regional governor's comments.

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
    rag_chain = create_retrieval_chain(retriever, document_chain)
    print("\nAnalist Asistanı hazır.")
    print("="*50)
    return rag_chain

# Bu dosya doğrudan çalıştırıldığında bir şey yapmaması için if __name__ bloğunu boş bırakabilir veya kaldırabiliriz.
if __name__ == '__main__':
    print("Bu dosya bir kütüphanedir ve doğrudan çalıştırılmak için tasarlanmamıştır.")
    print("Lütfen 'main_controller.py' dosyasını çalıştırın.")