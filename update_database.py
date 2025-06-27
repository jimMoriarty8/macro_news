# update_database.py (GÜNCELLENMİŞ HALİ)

import pandas as pd
import os
import html
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import config  # <-- YENİ: config.py import edildi

load_dotenv()

# --- AYARLAR ---
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
KNOWLEDGE_BASE_CSV = config.KNOWLEDGE_BASE_CSV
CHROMA_DB_PATH = config.CHROMA_DB_PATH
HEDEF_SEMBOLLER = config.SYMBOLS_TO_TRACK

def update_knowledge_base():
    print("\nHafıza güncelleme süreci başlatıldı...")

    # --- YENİ: TÜM YENİ VERİLERİ (CANLI + TOPLU) BİRLEŞTİRME ---
    all_new_data = []
    
    # 1. Canlı buffer dosyasını oku (varsa)
    if os.path.exists(config.LIVE_BUFFER_CSV):
        try:
            df_live = pd.read_csv(config.LIVE_BUFFER_CSV)
            all_new_data.append(df_live)
            print(f"'{config.LIVE_BUFFER_CSV}' dosyasından {len(df_live)} canlı haber okundu.")
        except pd.errors.EmptyDataError: pass

    # 2. Toplu veri dosyasını oku (varsa)
    if os.path.exists(config.RAW_NEWS_CSV):
        try:
            df_raw = pd.read_csv(config.RAW_NEWS_CSV)
            all_new_data.append(df_raw)
            print(f"'{config.RAW_NEWS_CSV}' dosyasından {len(df_raw)} toplu haber okundu.")
        except pd.errors.EmptyDataError: pass
    
    if not all_new_data:
        print("İşlenecek yeni haber bulunamadı. İşlem sonlandırılıyor.")
        return

    df_new_raw = pd.concat(all_new_data, ignore_index=True)
    
    if os.path.exists(config.LIVE_BUFFER_CSV): os.remove(config.LIVE_BUFFER_CSV)
    if os.path.exists(config.RAW_NEWS_CSV): os.remove(config.RAW_NEWS_CSV)
    print("Buffer dosyaları temizlendi.")
    # --- BİRLEŞTİRME SONU ---
    
    df_new_filtered = df_new_raw # Devam eden kodun uyumlu olması için yeniden isimlendirme
    
    # ... Buradan sonrası önceki adımdaki temizleme, sıralama ve kaydetme kodlarınızla aynı ...
    # (Mükerrer kayıtları silme, vektör db'yi güncelleme ve ana arşivi sıralayıp kaydetme)
    print(f"{len(df_new_filtered)} adet yeni haber işlenecek.")
    df_new_filtered['rag_content'] = df_new_filtered.apply(
        lambda row: f"{row['headline']}. {row['summary']}" if pd.notna(row['summary']) and len(str(row['summary']).split()) > 5 else row['headline'],
        axis=1
    )
    
    # ... (Vektör veritabanı güncelleme kodları) ...
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    new_documents = [ Document(page_content=row['rag_content'], metadata={'source': row.get('source', 'N/A'), 'title': row.get('headline', 'N/A'), 'publish_date': row.get('timestamp', 'N/A')}) for index, row in df_new_filtered.iterrows()]
    if new_documents:
        vector_store.add_documents(new_documents)
        print(f"{len(new_documents)} yeni vektör eklendi.")

    # ... (Ana arşivi birleştirme, sıralama ve kaydetme kodları) ...
    try:
        df_existing = pd.read_csv(KNOWLEDGE_BASE_CSV)
    except FileNotFoundError:
        df_existing = pd.DataFrame()
    df_combined = pd.concat([df_existing, df_new_filtered], ignore_index=True)
    df_combined.drop_duplicates(subset=['timestamp', 'headline'], keep='first', inplace=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.sort_values(by='timestamp', ascending=False, inplace=True)
    df_combined.to_csv(KNOWLEDGE_BASE_CSV, index=False, encoding='utf-8-sig')
    
    print(f"Ana arşiv '{KNOWLEDGE_BASE_CSV}' güncellendi. Toplam haber: {len(df_combined)}")
    print("\nTüm güncelleme işlemleri tamamlandı!")

if __name__ == '__main__':
    update_knowledge_base()