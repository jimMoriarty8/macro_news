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
LIVE_BUFFER_CSV = config.LIVE_BUFFER_CSV
EMBEDDING_MODEL = config.EMBEDDING_MODEL
# --- AYARLAR SONU ---

def update_and_build_databases():
    """
    Geçmiş (`temp_raw_news.csv`) ve canlı (`live_buffer.csv`) haber kaynaklarını okur,
    işler ve hem ana CSV arşivini hem de ChromaDB vektör veritabanını günceller.
    """
    print("\nVeritabanı oluşturma/güncelleme süreci başlatıldı...")

    # 1. Tüm yeni veri kaynaklarını (geçmiş ve canlı) topla
    dfs_to_process = []
    files_to_clean = []

    if os.path.exists(RAW_DATA_CSV):
        try:
            df_raw = pd.read_csv(RAW_DATA_CSV)
            if not df_raw.empty:
                print(f"'{RAW_DATA_CSV}' dosyasından {len(df_raw)} geçmiş haber yüklendi.")
                dfs_to_process.append(df_raw)
                files_to_clean.append(RAW_DATA_CSV)
        except pd.errors.EmptyDataError:
            print(f"UYARI: '{RAW_DATA_CSV}' dosyası boş.")

    if os.path.exists(LIVE_BUFFER_CSV):
        try:
            df_live = pd.read_csv(LIVE_BUFFER_CSV)
            if not df_live.empty:
                print(f"'{LIVE_BUFFER_CSV}' dosyasından {len(df_live)} canlı haber yüklendi.")
                dfs_to_process.append(df_live)
                files_to_clean.append(LIVE_BUFFER_CSV)
        except pd.errors.EmptyDataError:
            print(f"UYARI: '{LIVE_BUFFER_CSV}' dosyası boş.")

    if not dfs_to_process:
        print("İşlenecek yeni haber bulunamadı. İşlem durduruluyor.")
        return

    # Tüm yeni verileri tek bir DataFrame'de birleştir
    df_new_raw = pd.concat(dfs_to_process, ignore_index=True)
    # Olası mükerrer kayıtları (aynı anda hem geçmişten hem canlıdan gelmiş olabilir) temizle
    df_new_raw.drop_duplicates(subset=['id'], inplace=True)

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

    print("LangChain dökümanları hazırlanıyor...")
    documents_to_embed = [
        Document(
            # Boş içerik durumunda çökmemesi için varsayılan bir metin sağlıyoruz.
            page_content=str(row['rag_content']) if pd.notna(row['rag_content']) else "Content not available",
            metadata={
                'source': row.get('source', 'N/A'), 
                'title': row.get('headline', 'N/A'), 
                'publish_date': str(row.get('timestamp', 'N/A'))
            }
        )
        for _, row in tqdm(df_combined.iterrows(), total=df_combined.shape[0], desc="Dökümanlar Vektöre Çevriliyor")
    ]
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
    # --- KOD SAĞLAMLAŞTIRMA ---
    # Veritabanının diske tam olarak yazıldığından emin olmak için persist() metodunu çağırıyoruz.
    print("Veritabanı diske kaydediliyor...")
    db.persist()
    db = None # Belleği serbest bırak
    
    # 5. İşlenen geçici dosyaları temizle
    print("\nİşlenen geçici dosyalar temizleniyor...")
    for file_path in files_to_clean:
        try:
            os.remove(file_path)
            print(f" - '{os.path.basename(file_path)}' silindi.")
        except OSError as e:
            print(f"HATA: '{os.path.basename(file_path)}' silinirken hata oluştu: {e}")

    print("ChromaDB başarıyla oluşturuldu ve veriler kalıcı olarak kaydedildi.")
    print("\nTüm veri işleme işlemleri tamamlandı.")

if __name__ == '__main__':
    update_and_build_databases()