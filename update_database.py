# update_database.py

# update_database.py dosyasının en üstüne
from langchain_community.vectorstores.utils import filter_complex_metadata
import pandas as pd
import os
import html
import shutil
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from tqdm import tqdm
import config

# .env dosyasındaki API anahtarlarını yükle
load_dotenv()

# --- AYARLAR ---
# Ayarların hepsi merkezi config dosyamızdan geliyor
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY") # Bu satır aslında analysis_engine'da gerekli ama burada olması da zarar vermez.
RAW_DATA_CSV = config.RAW_NEWS_CSV
KNOWLEDGE_BASE_CSV = config.KNOWLEDGE_BASE_CSV
CHROMA_DB_PATH = config.CHROMA_DB_PATH
EMBEDDING_MODEL = config.EMBEDDING_MODEL
# --- AYARLAR SONU ---

def update_and_build_databases():
    """
    Ham veri dosyasını okur, işler ve hem ana CSV arşivini hem de
    ChromaDB vektör veritabanını sıfırdan oluşturur/günceller.
    """
    print("\nVeritabanı oluşturma/güncelleme süreci başlatıldı...")

    # 1. Ham veriyi yükle
    if not os.path.exists(RAW_DATA_CSV):
        print(f"HATA: '{RAW_DATA_CSV}' dosyası bulunamadı. Görünüşe göre collect_data.py hiç haber bulamadı. İşlem durduruluyor.")
        return
    
    try:
        df_new_raw = pd.read_csv(RAW_DATA_CSV)
        if df_new_raw.empty:
            print(f"UYARI: '{RAW_DATA_CSV}' dosyası boş. İşlenecek yeni haber yok. İşlem durduruluyor.")
            return
        print(f"'{RAW_DATA_CSV}' dosyasından {len(df_new_raw)} ham haber yüklendi.")
    except pd.errors.EmptyDataError:
        print(f"UYARI: '{RAW_DATA_CSV}' dosyası boş. İşlenecek yeni haber yok. İşlem durduruluyor.")
        return

    # 2. Veriyi temizle ve işle
    print("Veriler temizleniyor ve RAG formatına getiriliyor...")
    df_new_raw['headline'] = df_new_raw['headline'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
    df_new_raw['summary'] = df_new_raw['summary'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
    df_new_raw.dropna(subset=['id', 'headline'], inplace=True)
    df_new_raw['symbols'] = df_new_raw['symbols'].astype(str)

    def create_rag_content(row):
        headline = row['headline']
        summary = row['summary']
        if pd.notna(summary) and len(str(summary).split()) > 5:
            return f"Headline: {headline}. Summary: {summary}"
        return headline
    df_new_raw['rag_content'] = df_new_raw.apply(create_rag_content, axis=1)

    # 3. Ana arşivi (`knowledge_base.csv`) oluştur/güncelle
    try:
        df_existing_kb = pd.read_csv(KNOWLEDGE_BASE_CSV)
    except FileNotFoundError:
        df_existing_kb = pd.DataFrame()
    
    df_combined = pd.concat([df_existing_kb, df_new_raw], ignore_index=True)
    df_combined.drop_duplicates(subset=['id'], keep='last', inplace=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.sort_values(by='timestamp', ascending=False, inplace=True)
    
    df_combined.to_csv(KNOWLEDGE_BASE_CSV, index=False, encoding='utf-8-sig')
    print(f"Ana arşiv '{KNOWLEDGE_BASE_CSV}' güncellendi. Toplam haber sayısı: {len(df_combined)}")

    # 4. ChromaDB'yi sıfırdan oluştur
    print("\nMevcut ChromaDB (varsa) siliniyor ve temiz veriden yeniden oluşturuluyor...")
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    # --- KALICI DÜZELTME: VERİ TEMİZLİĞİ ---
    # Veritabanı oluşturmadan önce 'rag_content' içeriği boş olan satırları temizle.
    df_combined.dropna(subset=['rag_content'], inplace=True)
    print(f"Boş 'rag_content' satırları temizlendi. Vektör veritabanına eklenecek haber sayısı: {len(df_combined)}")

    print("LangChain dökümanları hazırlanıyor...")
    
    documents_to_embed = [
        Document(
            page_content=row['rag_content'],
            metadata={'source': row.get('source'), 'title': row.get('headline'), 'publish_date': row.get('timestamp')}
        )
        for _, row in tqdm(df_combined.iterrows(), total=df_combined.shape[0], desc="Dökümanlar Vektöre Çevriliyor")
    ]
    # --- YENİ EKLENEN BÖLÜM ---
    print("Karmaşık metadata (tarih formatı gibi) temizleniyor...")
    documents_to_embed = filter_complex_metadata(documents_to_embed)
    if not documents_to_embed:
        print("Vektör veritabanına eklenecek döküman bulunamadı.")
        return

    print("Embedding modeli ve ChromaDB başlatılıyor...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    print(f"{len(documents_to_embed)} döküman ChromaDB'ye ekleniyor. Bu işlem biraz sürebilir...")
    db = Chroma.from_documents(
        documents=documents_to_embed,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    
    print("ChromaDB başarıyla oluşturuldu ve veriler kalıcı olarak kaydedildi.")
    print("\nTüm veri işleme işlemleri tamamlandı.")

if __name__ == '__main__':
    update_and_build_databases()