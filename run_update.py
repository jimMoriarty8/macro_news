from collect_data import collect_historical_news
from update_database import update_knowledge_base
import time

def main():
    """
    Sistemin hafızasını güncellemek için tüm adımları sırayla çalıştırır.
    """
    start_time = time.time()
    
    print("="*50)
    print("SİSTEM HAFIZA GÜNCELLEME SÜRECİ BAŞLATILDI")
    print("="*50)
    
    # 1. Adım: Alpaca'dan yeni haberleri topla.
    collect_historical_news()
    
    # 2. Adım: Yeni haberlerle veritabanlarını güncelle.
    update_knowledge_base()
    
    end_time = time.time()
    print("\n" + "="*50)
    print(f"TÜM GÜNCELLEME İŞLEMLERİ TAMAMLANDI. Toplam Süre: {end_time - start_time:.2f} saniye.")
    print("="*50)

if __name__ == '__main__':
    main()