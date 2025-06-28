# collect_data.py (GEÇİCİ TEST KODU)

import os
from dotenv import load_dotenv
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest
import config

# .env dosyasındaki API anahtarlarını yükle
load_dotenv()

# --- AYARLAR ---
API_KEY = os.getenv(config.ALPACA_API_KEY_ENV)
SECRET_KEY = os.getenv(config.ALPACA_SECRET_KEY_ENV)
news_client = NewsClient(API_KEY, SECRET_KEY)

def run_test_query():
    """Sadece tek bir sorgu göndererek API cevabını test eder."""
    print("--- Alpaca API Testi Başlatıldı ---")
    
    # Sadece çok bilinen tek bir sembol ve yakın bir tarih aralığı ile test edelim
    test_symbols = "BTC/USD"
    test_start_date = "2024-06-01"
    test_end_date = "2024-06-02"
    
    print(f"Sorgulanan Semboller: {test_symbols}")
    print(f"Sorgulanan Tarih Aralığı: {test_start_date} - {test_end_date}")
    
    try:
        request_params = NewsRequest(
            symbols=test_symbols,
            start=test_start_date,
            end=test_end_date,
            limit=10 # Sadece birkaç sonuç yeterli
        )
        
        # Alpaca'dan gelen ham cevabı doğrudan yazdır
        news_response = news_client.get_news(request_params)
        
        print("\n--- ALPACA'DAN GELEN HAM CEVAP ---")
        print(news_response)
        print("------------------------------------")

        # Cevabın içini kontrol et
        if news_response and hasattr(news_response, 'news') and news_response.news:
            print(f"\nBAŞARILI! {len(news_response.news)} adet haber bulundu.")
        else:
            print("\nBAŞARISIZ! Cevap alındı ama içinde haber verisi yok.")

    except Exception as e:
        print(f"\n!!! API İSTEĞİ SIRASINDA HATA OLUŞTU: {e}")

if __name__ == '__main__':
    run_test_query()