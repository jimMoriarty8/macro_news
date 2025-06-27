# main.py (GÜNCELLENMİŞ HALİ)

import os
import html
import logging
from dotenv import load_dotenv
from alpaca.data.live.news import NewsDataStream
import pandas as pd  # <-- YENİ: Pandas import edildi

# Kendi dosyalarımızdan importlar
import config  # <-- YENİ: config.py import edildi
from analysis_engine import (
    initialize_analyst_assistant, 
    parse_analyst_report, 
    send_telegram_message, 
    get_btc_price
)

load_dotenv()

# --- 1. AYARLAR ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
CONFIDENCE_THRESHOLD = 7
IMPACT_THRESHOLD = 7
SYMBOL_WATCHLIST = config.SYMBOLS_TO_TRACK  # <-- DEĞİŞTİ: Ayar config'den geliyor

# --- 2. KONTROL MERKEZİ ---
retriever, document_chain = initialize_analyst_assistant()

async def analyze_news_on_arrival(data):
    """
    Alpaca'dan gelen haberi alır, analiz eder, alarm gönderir VE GEÇİCİ DOSYAYA KAYDEDER.
    """
    # --- YENİ: CANLI HABERİ DOSYAYA KAYDETME ---
    try:
        news_data = {
            "id": [data.id], "timestamp": [data.created_at], "headline": [data.headline],
            "summary": [data.summary], "source": [data.source], 
            "symbols": [",".join(data.symbols) if data.symbols else ""]
        }
        df_live = pd.DataFrame(news_data)
        
        header_needed = not os.path.exists(config.LIVE_BUFFER_CSV)
        df_live.to_csv(config.LIVE_BUFFER_CSV, mode='a', header=header_needed, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"HATA: Canlı haber buffer'a kaydedilemedi: {e}")
    # --- KAYDETME BÖLÜMÜ SONU ---

    try:
        is_relevant = any(
            watched_symbol in news_symbol
            for news_symbol in data.symbols
            for watched_symbol in SYMBOL_WATCHLIST
        )
        if not is_relevant:
            return

        headline = html.unescape(data.headline)
        print(f"\n📰 [İLGİLİ HABER GELDİ] {headline}")
        print("   Analiz ediliyor...")

        retrieved_docs = retriever.get_relevant_documents(headline)
        retrieved_docs.sort(key=lambda x: x.metadata.get('publish_date', '1970-01-01'))

        report_text = document_chain.invoke({
            "input": headline,
            "context": retrieved_docs
        })
        
        print("\n--- ANALYST REPORT ---")
        print(report_text)
        
        parsed_report = parse_analyst_report(report_text)
        
        if parsed_report:
            direction = parsed_report.get('direction', 'N/A')
            impact = parsed_report.get('impact', 0)
            confidence = parsed_report.get('confidence', 0)

            print(f"\n--- Karar Motoru ---")
            print(f"Tespit Edilen Yön: {direction}, Etki: {impact}, Güven: {confidence}")

            if direction.lower() != 'neutral' and confidence >= CONFIDENCE_THRESHOLD and impact >= IMPACT_THRESHOLD:
                print(f"✅ ALARM KRİTERLERİ KARŞILANDI!")
                btc_price = get_btc_price()
                direction_emoji = "🟢" if direction.lower() == 'positive' else "🔴" if direction.lower() == 'negative' else "⚪️"
                
                message = (
                    f"{direction_emoji} *High-Potential Signal: {direction.upper()}*\n"
                    f"*BTC/USDT Price:* `{btc_price}`\n\n"
                    f"*Headline:*\n`{headline}`\n\n"
                    f"*Scores:*\n"
                    f"Impact: *{impact}/10* | Confidence: *{confidence}/10*\n\n"
                    f"*Analyst Comment:*\n_{parsed_report.get('analysis', '')}_"
                )
                send_telegram_message(message)
            else:
                print("❌ Alarm kriterleri karşılanmadı.")
        
        print("="*50)

    except Exception as e:
        print(f"\n🚨 ANALİZ SIRASINDA KRİTİK BİR HATA OLUŞTU: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    news_stream.subscribe_news(analyze_news_on_arrival, '*')
    print(f"--- CANLI HABER ANALİZ SİSTEMİ AKTİF ---")
    print(f"İzleme Listesi: {SYMBOL_WATCHLIST}")
    news_stream.run()