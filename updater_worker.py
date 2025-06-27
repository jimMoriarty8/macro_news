# updater_worker.py

import schedule
import time
from run_update import main as run_update_task # run_update.py'deki ana fonksiyonumuzu import ediyoruz

def job():
    """
    Zamanlanmış güncelleme görevini çalıştıran fonksiyon.
    """
    print(f"[{time.ctime()}] Zamanlanmış veritabanı güncelleme görevi başlatılıyor...")
    try:
        run_update_task()
    except Exception as e:
        print(f"!!! GÜNCELLEME SIRASINDA KRİTİK HATA: {e}")
    print(f"[{time.ctime()}] Görev tamamlandı. Bir sonraki çalıştırma bekleniyor.")

# --- ZAMANLAMA KURALI ---
# Her gün, sunucunun yerel saatiyle 03:00'da 'job' fonksiyonunu çalıştıracak şekilde ayarla.
# Render sunucuları UTC saat dilimini kullanır.
schedule.every().day.at("03:00").do(job)

print("UPDATER WORKER BAŞLATILDI - Görevler zamanlandı. İlk çalıştırma UTC 03:00'da.")

# Sonsuz döngü: Her 60 saniyede bir, "çalıştırılması gereken bir görev var mı?" diye kontrol et.
while True:
    schedule.run_pending()
    time.sleep(60)