# main.py

import os
import html
import logging
import pandas as pd
import asyncio
from dotenv import load_dotenv
from alpaca.data.live.news import NewsDataStream
from langchain.docstore.document import Document
from langchain_chroma import Chroma

# Kendi dosyalarımızdan importlar
import config
from analysis_engine import (
    initialize_analyst_assistant, 
    parse_analyst_report, 
    send_telegram_message, 
    get_btc_price
)

# .env dosyasındaki tüm değişkenleri yükle
load_dotenv()

# --- 1. AYARLAR ---
ALPACA_API_KEY = os.getenv(config.ALPACA_API_KEY_ENV)
ALPACA_SECRET_KEY = os.getenv(config.ALPACA_SECRET_KEY_ENV)
CONFIDENCE_THRESHOLD = 7
IMPACT_THRESHOLD = 7
SYMBOL_WATCHLIST = config.SYMBOLS_TO_TRACK

# --- 2. RAG SİSTEMİ KURULUMU ---
# Program başlarken, analiz ve güncelleme için gerekli olan 3 aracı birden alıyoruz.
retriever, document_chain, vector_store = initialize_analyst_assistant()

# --- 3. ARKA PLAN VERİTABANI GÜNCELLEME GÖREVİ ("ŞEF") ---
async def process_and_save_in_background(news_dict: dict, vs: Chroma):
    """
    Bu fonksiyon arka planda çalışır, ana botu bloklamaz.
    Tek bir haberi işler ve veritabanlarına ekler.
    """
    try:
        headline_short = news_dict.get('headline', '')[:40]
        print(f"   -> Arka plan görevi başladı: '{headline_short}...' veritabanına ekleniyor.")
        
        # 1. Haberi işle ve RAG formatına getir
        df_item = pd.DataFrame(news_dict, index=[0])
        headline = df_item.iloc[0]['headline']
        summary = df_item.iloc[0]['summary']
        rag_content = f"Headline: {headline}. Summary: {summary}" if pd.notna(summary) and len(str(summary).split()) > 5 else headline
        
        # 2. Knowledge Base CSV'ye ekle (append modunda)
        header_needed = not os.path.exists(config.KNOWLEDGE_BASE_CSV)
        # Bu I/O işlemini de ana döngüyü bloklamamak için thread'de çalıştırabiliriz.
        await asyncio.to_thread(
            df_item.to_csv, config.KNOWLEDGE_BASE_CSV, mode='a', 
            header=header_needed, index=False, encoding='utf-8-sig'
        )

        # 3. ChromaDB'ye eklemek için Document nesnesi oluştur
        doc_to_add = Document(
            page_content=rag_content,
            metadata={'source': df_item.iloc[0]['source'], 'title': headline, 'publish_date': df_item.iloc[0]['timestamp']}
        )
        
        # 4. ChromaDB'ye ekle (Bu senkron bir işlem olduğu için thread'de çalıştırarak ana döngüyü bloklamasını önle)
        await asyncio.to_thread(vs.add_documents, [doc_to_add])
        
        print(f"   -> Arka plan görevi bitti: '{headline_short}...' eklendi.")
    except Exception as e:
        print(f"!!! ARKA PLAN GÜNCELLEME HATASI: {e}")


# --- 4. CANLI HABER ANALİZ FONKSİYONU ("GARSON") ---
async def analyze_news_on_arrival(data):
    """
    Hızlıca haberi alır, analiz eder ve yavaş olan kaydetme işini arka plana atar.
    """
    try:
        # İlgililik kontrolü
        is_relevant = any(watched_symbol in str(data.symbols) for watched_symbol in SYMBOL_WATCHLIST)
        if not is_relevant:
            return

        headline = html.unescape(data.headline)
        print(f"\n📰 [İLGİLİ HABER GELDİ] {headline}")

        # Hızlı Analiz ve Alarm Kısmı
        print("   Analiz ediliyor...")
        retrieved_docs = retriever.get_relevant_documents(headline)
        retrieved_docs.sort(key=lambda x: x.metadata.get('publish_date', '1970-01-01'))
        report_text = document_chain.invoke({"input": headline, "context": retrieved_docs})
        
        print("\n--- ANALYST REPORT ---")
        print(report_text)
        
        parsed_report = parse_analyst_report(report_text)
        if parsed_report and parsed_report.get('confidence', 0) >= CONFIDENCE_THRESHOLD and parsed_report.get('impact', 0) >= IMPACT_THRESHOLD and parsed_report.get('direction').lower() != 'neutral':
            print(f"✅ ALARM KRİTERLERİ KARŞILANDI!")
            btc_price = get_btc_price()
            direction = parsed_report.get('direction', 'N/A')
            direction_emoji = "🟢" if direction.lower() == 'positive' else "🔴" if direction.lower() == 'negative' else "⚪️"
            message = (f"{direction_emoji} *High-Potential Signal: {direction.upper()}*\n" f"*BTC/USDT Price:* `{btc_price}`\n\n" f"*Headline:*\n`{headline}`\n\n" f"*Scores:*\n" f"Impact: *{parsed_report.get('impact')}/10* | Confidence: *{parsed_report.get('confidence')}/10*\n\n" f"*Analyst Comment:*\n_{parsed_report.get('analysis', '')}_")
            send_telegram_message(message)
        else:
            print("❌ Alarm kriterleri karşılanmadı veya yön 'Neutral'.")
        
        # --- YAVAŞ İŞİ ARKA PLANA ATMA ---
        # Analiz bittikten sonra, veritabanı kaydetme işini bir arka plan görevine devret ve bekleme yapma.
        news_item_dict = {"id": data.id, "timestamp": data.created_at, "headline": data.headline, "summary": data.summary, "source": data.source, "symbols": ",".join(data.symbols) if data.symbols else ""}
        asyncio.create_task(process_and_save_in_background(news_item_dict, vector_store))

    except Exception as e:
        print(f"\n🚨 ANA ANALİZ DÖNGÜSÜ HATASI: {e}")


# --- 5. ANA UYGULAMAYI BAŞLATMA ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    news_stream.subscribe_news(analyze_news_on_arrival, '*')
    print(f"--- CANLI HABER ANALİZ SİSTEMİ AKTİF (ANLIK ÖĞRENME MODU) ---")
    news_stream.run()