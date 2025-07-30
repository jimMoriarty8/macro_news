# run_update.py


# Diğer betiklerimizdeki ana fonksiyonları import ediyoruz
from collect_data import collect_historical_news
from update_database import update_and_build_databases
import time

def main():
    """
    Tüm veri toplama ve veritabanı oluşturma sürecini baştan sona yönetir.
    """
    start_time = time.time()
    
    print("="*50)
    print("ORKESTRA ŞEFİ: Geçmiş Veri Toplama ve Veritabanı Kurulum Süreci Başlatıldı")
    print("="*50)
    
    # 1. Adım: Alpaca'dan geçmiş verileri topla ve geçici dosyaya yaz.
    print("\n--- ADIM 1: Veri Toplama Başlatılıyor ---")
    collect_historical_news()
    
    # 2. Adım: Toplanan verileri işle ve ana veritabanlarını oluştur/güncelle.
    print("\n--- ADIM 2: Veritabanı Oluşturma/Güncelleme Başlatılıyor ---")
    update_and_build_databases()
    
    end_time = time.time()
    print("\n" + "="*50)
    print(f"TÜM GEÇMİŞ VERİ İŞLEMLERİ TAMAMLANDI. Toplam Süre: {end_time - start_time:.2f} saniye.")
    print("="*50)

if __name__ == '__main__':
    main()