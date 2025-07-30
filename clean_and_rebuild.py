# clean_and_rebuild.py


import pandas as pd
import os
import shutil
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from tqdm import tqdm # İlerleme çubuğu için güzel bir kütüphane (pip install tqdm)

# .env dosyasını yükle
load_dotenv()

# --- AYARLAR ---
# Bu ayarların diğer betiklerinizdekiyle aynı olduğundan emin olun
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
KNOWLEDGE_BASE_CSV = "knowledge_base.csv"
CHROMA_DB_PATH = "./chroma_db"
# --- AYARLAR SONU ---

def clean_and_rebuild_all():
    """
    Tüm veritabanını temizler ve yeniden inşa eder.
    1. knowledge_base.csv'deki mükerrer kayıtları siler.
    2. Eski chroma_db klasörünü siler.
    3. Temiz CSV'den yeni bir chroma_db oluşturur.
    """
    print("="*50)
    print("VERİTABANI TEMİZLEME VE YENİDEN OLUŞTURMA SÜRECİ")
    print("="*50)

    # --- 1. ADIM: KNOWLEDGE_BASE.CSV'Yİ TEMİZLE ---
    print(f"\n--- Adım 1: '{KNOWLEDGE_BASE_CSV}' temizleniyor... ---")
    try:
        df = pd.read_csv(KNOWLEDGE_BASE_CSV)
        initial_rows = len(df)
        print(f"Başlangıçtaki satır sayısı: {initial_rows}")

        # 'timestamp' ve 'title' sütunlarına göre mükerrer olanları bul ve ilkini tut
        df.drop_duplicates(subset=['timestamp', 'title'], keep='first', inplace=True)
        
        # Tarihe göre yeniden sırala (isteğe bağlı ama düzenli tutar)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', ascending=False, inplace=True)
        
        final_rows = len(df)
        removed_rows = initial_rows - final_rows

        # Temizlenmiş veriyi aynı dosyanın üzerine yaz
        df.to_csv(KNOWLEDGE_BASE_CSV, index=False, encoding='utf-8-sig')
        
        print(f"Bitişteki satır sayısı: {final_rows}")
        print(f"✅ {removed_rows} adet mükerrer kayıt silindi. '{KNOWLEDGE_BASE_CSV}' başarıyla temizlendi.")

    except FileNotFoundError:
        print(f"HATA: '{KNOWLEDGE_BASE_CSV}' dosyası bulunamadı. İşlem durduruldu.")
        return
    except Exception as e:
        print(f"CSV temizlenirken bir hata oluştu: {e}")
        return

    # --- 2. ADIM: ESKİ CHROMA_DB'Yİ SİL ---
    print(f"\n--- Adım 2: Eski '{CHROMA_DB_PATH}' klasörü siliniyor... ---")
    if os.path.exists(CHROMA_DB_PATH):
        try:
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"✅ '{CHROMA_DB_PATH}' klasörü başarıyla silindi.")
        except Exception as e:
            print(f"Klasör silinirken bir hata oluştu: {e}")
            return
    else:
        print("Eski veritabanı klasörü bulunamadı, silme adımı atlanıyor.")

    # --- 3. ADIM: TEMİZ CSV'DEN YENİ CHROMA_DB OLUŞTUR ---
    print(f"\n--- Adım 3: Temiz veriden yeni bir vektör veritabanı oluşturuluyor... ---")
    print("Bu işlem veri miktarına göre biraz zaman alabilir.")
    try:
        # Temizlenmiş CSV'yi yeniden oku
        df_clean = pd.read_csv(KNOWLEDGE_BASE_CSV)
        
        # Verileri LangChain Document formatına çevir
        documents = [
            Document(
                page_content=row['rag_content'],
                metadata={'source': row.get('source', 'N/A'), 
                          'title': row.get('title', 'N/A'), 
                          'publish_date': row.get('timestamp', 'N/A')}
            ) 
            for index, row in tqdm(df_clean.iterrows(), total=df_clean.shape[0], desc="Dökümanlar işleniyor")
        ]

        # Vektör oluşturucu (embedding model)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Sıfırdan veritabanı oluştur
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"\n✅ Yeni ve temiz vektör veritabanı '{CHROMA_DB_PATH}' klasöründe başarıyla oluşturuldu!")

    except Exception as e:
        print(f"Yeni veritabanı oluşturulurken bir hata oluştu: {e}")
        return
        
    print("\n" + "="*50)
    print("TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
    print("="*50)


if __name__ == '__main__':
    # Gerekli kütüphaneyi yüklemek için: pip install tqdm
    clean_and_rebuild_all()