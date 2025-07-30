
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv
import config
from alpaca.data.historical import NewsClient
from alpaca.data.requests import NewsRequest

load_dotenv()

# --- 1. AYARLAR ---
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
news_client = NewsClient(API_KEY, SECRET_KEY)

CSV_FILENAME = config.RAW_NEWS_CSV
# Ayarları artık merkezi config dosyasından alıyoruz. Bu, tutarlılığı sağlar ve hataları önler.
BASLANGIC_TARIHI = config.ARCHIVE_START_DATE
SEMBOLLER_LISTESI = config.SYMBOLS_TO_TRACK
SEMBOLLER_STRING = ",".join(SEMBOLLER_LISTESI)
# --- AYARLAR SONU ---

def collect_historical_news():
    print("Alpaca Arşivleme Script'i Başlatıldı...")
    
    if os.path.exists(CSV_FILENAME):
        try:
            df_existing = pd.read_csv(CSV_FILENAME)
            cekilen_haber_idleri = set(df_existing['id']) if 'id' in df_existing.columns else set()
            print(f"Mevcut arşivde {len(df_existing)} haber bulundu.")
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame()
            cekilen_haber_idleri = set()
            print("Mevcut arşiv dosyası boş.")
    else:
        df_existing = pd.DataFrame()
        cekilen_haber_idleri = set()

    tarih_araligi = pd.date_range(start=BASLANGIC_TARIHI, end=datetime.now(), freq='D')
    all_news_data = []

    print(f"{len(tarih_araligi)} gün için veri taranacak. Bu işlem uzun sürebilir...")

    for gun in reversed(tarih_araligi):
        print(f"\nİşlenen Tarih: {gun.date()}")
        page_token = None

        while True:
            try:
                request_params = NewsRequest(
                    symbols=SEMBOLLER_STRING,
                    start=gun.strftime("%Y-%m-%d"),
                    end=(gun + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    limit=50,
                    page_token=page_token
                )

                news_page = news_client.get_news(request_params)
                
                # --- KRİTİK DÜZELTME: DOĞRU VERİ ERİŞİM YOLU ---
                # alpaca-py kütüphanesi haber listesini doğrudan .news özelliği altında döndürür.
                if news_page and news_page.news:
                    haber_listesi = news_page.news
                    found_in_page = 0
                    for haber in haber_listesi:
                        if haber.id not in cekilen_haber_idleri:
                            all_news_data.append({
                                "id": haber.id,
                                "timestamp": haber.created_at,
                                "headline": haber.headline,
                                "summary": haber.summary,
                                "source": haber.source,
                                "symbols": haber.symbols
                            })
                            cekilen_haber_idleri.add(haber.id)
                            found_in_page += 1
                    
                    if found_in_page > 0:
                        print(f"  -> Sayfadan {found_in_page} yeni haber bulundu.")

                if news_page and news_page.next_page_token:
                    page_token = news_page.next_page_token
                    time.sleep(1) # Alpaca API limitlerine takılmamak için kısa bir bekleme
                else:
                    break
            
            except Exception as e:
                print(f"  -> HATA: {gun.date()} tarihinde veri çekilirken bir sorun oluştu: {e}")
                break

    if not all_news_data:
        print("\nArşive eklenecek yeni haber bulunamadı.")
        return

    print(f"\nToplam {len(all_news_data)} yeni haber toplandı.")

    df_new = pd.DataFrame(all_news_data)
    df_combined = pd.concat([df_existing, df_new]).drop_duplicates(subset=['id'], ignore_index=True)
    
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.sort_values(by='timestamp', ascending=False, inplace=True)
    
    df_combined.to_csv(CSV_FILENAME, index=False, encoding='utf-8-sig')
    print(f"\nİşlem tamamlandı. Arşivdeki toplam haber sayısı: {len(df_combined)}")
    print(f"Veriler '{CSV_FILENAME}' dosyasına kaydedildi.")

if __name__ == '__main__':
    collect_historical_news()