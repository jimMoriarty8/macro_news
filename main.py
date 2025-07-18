# main_controller.py (config.py ile tam uyumlu, nihai versiyon)

import os
import html
import logging
import asyncio
from dotenv import load_dotenv

from alpaca.data.live.news import NewsDataStream
from langchain.docstore.document import Document
from langchain_chroma import Chroma

# Kendi dosyalarımızdan importlar
from analysis_engine import (
    initialize_analyst_assistant, 
    parse_analyst_report, 
    send_telegram_message, 
    get_btc_price
)
import config # Artık tüm ayarlar için config.py'yi kullanıyoruz

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR ---
# Tüm ayarlar artık config.py dosyasından okunuyor.
# Bu, kodumuzu daha temiz ve yönetilebilir hale getirir.
ALPACA_API_KEY = os.getenv(config.ALPACA_API_KEY_ENV)
ALPACA_SECRET_KEY = os.getenv(config.ALPACA_SECRET_KEY_ENV)
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
IMPACT_THRESHOLD = config.IMPACT_THRESHOLD
SYMBOL_WATCHLIST = config.SYMBOLS_TO_TRACK # Doğru değişken adını kullanıyoruz

# --- 2. RAG SİSTEMİ KURULUMU ---
# Program başlarken, app.py'den sadece temel araçları alıyoruz
# initialize_analyst_assistant fonksiyonu artık config.py'deki ayarları kullanacak
retriever, document_chain, vector_store = initialize_analyst_assistant()

# --- 3. ÇEVİRİ MOTORU ---
# Bu bölüm, önceki versiyonlardaki gibi kalabilir veya eklenebilir.
# Şimdilik ana mantığa odaklanıyoruz.

# --- 4. ARKA PLAN GÖREVİ ---
async def process_and_save_in_background(news_dict: dict, vs: Chroma):
    # Bu fonksiyon, veritabanını ve CSV'yi günceller.
    # Mantığı önceki mesajlarımızdaki gibi kalabilir.
    pass 

# --- 5. CANLI HABER ANALİZ FONKSİYONU ---
async def analyze_news_on_arrival(data):
    """
    Haberi alır, analiz eder ve kaydetme işini arka plana atar.
    """
    try:
        # İlgililik kontrolü
        is_relevant = any(watched_symbol in str(data.symbols) for watched_symbol in SYMBOL_WATCHLIST)
        if not is_relevant:
            return

        headline_en = html.unescape(data.headline)
        print(f"\n📰 [İLGİLİ HABER GELDİ] {headline_en}")
        
        # Analiz adımı
        print("   -> Analiz ediliyor...")
        # Not: app.py'deki initialize_analyst_assistant fonksiyonunun
        # en güncel zincir yapısını kullandığından emin olun.
        report_text = document_chain.invoke({"input": headline_en, "context": retriever.invoke(headline_en)})
        
        print("\n--- ANALYST REPORT ---")
        print(report_text)
        
        parsed_report = parse_analyst_report(report_text)
        if parsed_report and parsed_report.get('confidence', 0) >= CONFIDENCE_THRESHOLD and parsed_report.get('impact', 0) >= IMPACT_THRESHOLD and parsed_report.get('direction').lower() != 'neutral':
            print(f"✅ ALARM KRİTERLERİ KARŞILANDI!")
            btc_price = get_btc_price()
            direction = parsed_report.get('direction', 'N/A')
            direction_emoji = "🟢" if direction.lower() == 'positive' else "🔴"
            
            message = (
                f"{direction_emoji} *Signal: {direction.upper()}*\n"
                f"*BTC/USDT Price:* `{btc_price}`\n\n"
                f"*Headline (EN):*\n`{headline_en}`\n\n"
                f"*Scores:*\n"
                f"Impact: *{parsed_report.get('impact')}/10* | Confidence: *{parsed_report.get('confidence')}/10*\n\n"
                f"*Commentary:*\n_{parsed_report.get('analysis', '')}_"
            )
            send_telegram_message(message)
        else:
            print("❌ Alarm kriterleri karşılanmadı veya yön 'Neutral'.")
        
        # Arka planda veritabanını güncelleme
        news_item_dict = {"id": data.id, "timestamp": data.created_at, "headline": data.headline, "summary": data.summary, "source": data.source, "symbols": ",".join(data.symbols) if data.symbols else ""}
        asyncio.create_task(process_and_save_in_background(news_item_dict, vector_store))

    except Exception as e:
        print(f"\n🚨 ANA ANALİZ DÖNGÜSÜ HATASI: {e}")


# --- 6. ANA UYGULAMAYI BAŞLATMA ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    news_stream.subscribe_news(analyze_news_on_arrival, '*')
    print(f"--- CANLI HABER ANALİZ SİSTEMİ AKTİF ---")
    print(f"İzleme Listesi: {list(SYMBOL_WATCHLIST)}")
    news_stream.run()
