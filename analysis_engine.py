# analysis_engine.py


import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config
import re
import requests
from binance.client import Client as BinanceClient
from deep_translator import GoogleTranslator

def initialize_analyst_components():
    """
    Vektör veritabanını, LLM'i ve retriever'ı kurar.
    Farklı amaçlar için yeniden kullanılabilecek temel RAG bileşenlerini döndürür.
    """
    print("RAG sistem bileşenleri başlatılıyor...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("HATA: GEMINI_API_KEY ortam değişkeni bulunamadı! Lütfen .env dosyasını veya Render ayarlarını kontrol edin.")

    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2, google_api_key=api_key)
    
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Mevcut vektör veritabanı '{config.CHROMA_DB_PATH}' klasöründen yükleniyor...")
        vector_store = Chroma(
            persist_directory=config.CHROMA_DB_PATH, 
            embedding_function=embeddings,
            # --- İYİLEŞTİRME: TELEMETRİYİ KAPATMA ---
            client_settings=Settings(anonymized_telemetry=False)
        )
        print("Veritabanı başarıyla yüklendi.")
    else:
        print(f"UYARI: Henüz bir veritabanı bulunamadı. '{config.KNOWLEDGE_BASE_CSV}' dosyasından oluşturulacak...")
        try:
            df = pd.read_csv(config.KNOWLEDGE_BASE_CSV)
            
            documents = [
                Document(
                    # Boş içerik durumunda çökmemesi için varsayılan bir metin sağlıyoruz.
                    page_content=str(row['rag_content']) if pd.notna(row['rag_content']) else "Content not available",
                    metadata={
                        'source': row.get('source', 'N/A'), 
                        'title': row.get('headline', 'N/A'), 
                        # Metadata'da tarih gibi karmaşık nesneler sorun çıkarabildiği için string'e çeviriyoruz.
                        'publish_date': str(row.get('timestamp', 'N/A'))
                    }
                ) 
                for index, row in df.iterrows()
            ]
            vector_store = Chroma.from_documents(
                documents=documents, 
                embedding=embeddings, 
                persist_directory=config.CHROMA_DB_PATH,
                client_settings=Settings(anonymized_telemetry=False)
            )
            print(f"Yeni veritabanı '{config.CHROMA_DB_PATH}' klasöründe başarıyla oluşturuldu.")
        except FileNotFoundError:
             print(f"HATA: '{config.KNOWLEDGE_BASE_CSV}' dosyası bulunamadı. Lütfen önce veri toplama ve işleme script'lerini çalıştırın.")
             exit()


    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    
    print("\nTemel RAG bileşenleri hazır.")
    print("="*50)
    return llm, retriever, vector_store

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

def translate_to_turkish(text: str) -> str:
    """Verilen metni Türkçe'ye çevirir."""
    if not text:
        return "Çeviri için metin mevcut değil."
    try:
        # deep-translator kütüphanesini kullanarak çeviri yapıyoruz.
        return GoogleTranslator(source='en', target='tr').translate(text)
    except Exception as e:
        print(f"UYARI: Metin çevrilemedi. Hata: {e}")
        return "Çeviri yapılamadı."

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
    # Bu fonksiyon, LLM'den gelen raporun formatındaki küçük değişikliklere karşı
    # daha dayanıklı ve verimli olacak şekilde iyileştirildi.
    try:
        # Tek bir derlenmiş regex ile tüm alanları yakalamak daha verimlidir.
        # re.DOTALL, '.' karakterinin yeni satırları da eşleştirmesini sağlar.
        pattern = re.compile(
            r"Direction:\s*\[?([^\]\n]+)\]?.*?"
            r"Impact Score:\s*\[?(\d+).*?"
            r"Confidence Score:\s*\[?(\d+).*?"
            r"Analysis:\s*(.*)",
            re.IGNORECASE | re.DOTALL
        )
        match = pattern.search(report_text)
        if not match:
            print(f"AYRIŞTIRMA HATASI: Rapor beklenen formatta değil. Rapor: {report_text[:200]}...")
            return None

        return {"direction": match.group(1).strip(), "impact": int(match.group(2)), "confidence": int(match.group(3)), "analysis": match.group(4).strip()}
    except (IndexError, ValueError) as e:
        print(f"AYRIŞTIRMA HATASI: Raporun bir kısmı ayrıştırılamadı. Hata: {e}")
        return None
