# collect_data.py (NİHAİ VE ÇALIŞAN HALİ)

import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
import config

load_dotenv()

# --- AYARLAR ---
API_KEY = os.getenv(config.ALPACA_API_KEY_ENV)
SECRET_KEY = os.getenv(config.ALPACA_SECRET_KEY_ENV)
news_client = NewsClient(API_KEY, SECRET_KEY)

# config'den gelen listeyi, Alpaca'nın istediği formata (virgülle ayrılmış metin) çeviriyoruz.
SYMBOLS_STRING = ",".join(config.SYMBOLS_TO_TRACK)
CSV_FILENAME = config.RAW_NEWS_CSV
START_DATE = config.ARCHIVE_START_DATE
# --- AYARLAR SONU ---

def collect_historical_news():
    print("Alpaca Geçmiş Veri Toplama Script'i Başlatıldı...")
    
    tarih_araligi = pd.date_range(start=START_DATE, end=datetime.now(), freq='D')
    all_news_data = []

    print(f"{len(tarih_araligi)} gün için veri taranacak. Bu işlem uzun sürebilir...")

    for gun in reversed(tarih_araligi):
        print(f"\nİşlenen Tarih: {gun.date()}", end="")
        page_token = None
        found_total_in_day = 0

        while True:
            try:
                request_params = NewsRequest(
                    symbols=SYMBOLS_STRING,
                    start=gun.strftime("%Y-%m-%d"),
                    end=(gun + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    limit=50,
                    page_token=page_token
                )
                news_page = news_client.get_news(request_params)
                
                if news_page and hasattr(news_page, 'news') and news_page.news:
                    haber_listesi = news_page.news
                    for haber in haber_listesi:
                        all_news_data.append({
                            "id": haber.id, "timestamp": haber.created_at, "headline": haber.headline,
                            "summary": haber.summary, "source": haber.source, "symbols": haber.symbols
                        })
                    found_total_in_day += len(haber_listesi)
                
                if news_page and news_page.next_page_token:
                    page_token = news_page.next_page_token
                    time.sleep(0.5)
                else:
                    break
            except Exception as e:
                print(f" -> HATA: {e}")
                break
        
        if found_total_in_day > 0:
            print(f" -> {found_total_in_day} ham haber bulundu.")

    if not all_news_data:
        print("\nİşlenecek yeni haber bulunamadı.")
        return

    print(f"\nToplam {len(all_news_data)} ham haber toplandı.")
    df_new = pd.DataFrame(all_news_data).drop_duplicates(subset=['id'], ignore_index=True)
    
    df_new.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print(f"Toplanan ham veriler geçici olarak '{CSV_FILENAME}' dosyasına kaydedildi.")

if __name__ == '__main__':
    collect_historical_news()