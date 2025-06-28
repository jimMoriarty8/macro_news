# update_database.py (BASİTLEŞTİRİLMİŞ HALİ)

import pandas as pd
import os
import html
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

# .env dosyasını yükle
load_dotenv()

# --- AYARLAR (DEĞİŞİKLİK YOK) ---
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

NEW_DATA_CSV = "temp_raw_news.csv" 
KNOWLEDGE_BASE_CSV = "knowledge_base.csv"
CHROMA_DB_PATH = "./chroma_db"

HEDEF_SEMBOLLER = ['BTC/USD', "BTC", 'BTCUSD', 'ETH/USD', "ETH", 'ETHUSD',
                   "SOL/USD", "SOL", 'SOLUSD', "XRP/USD", "XRP", 'XRPUSD',
                   "BNB/USD", "BNB", 'BNBUSD', 'SPY', 'QQQ']
# --- AYARLAR SONU ---

def update_knowledge_base():
    print("Hafıza güncelleme süreci başlatıldı...")

    # --- Adım 1: Yeni Veriyi Yükle ve İşle (DEĞİŞİKLİK YOK) ---
    try:
        df_new = pd.read_csv(NEW_DATA_CSV)
        print(f"'{NEW_DATA_CSV}' dosyasından {len(df_new)} yeni haber yüklendi.")
    except FileNotFoundError:
        print(f"HATA: '{NEW_DATA_CSV}' dosyası bulunamadı. Lütfen yeni haberleri bu isimle kaydedin.")
        return

    print("Yeni veriler temizleniyor ve RAG formatına getiriliyor...")
    df_new['headline'] = df_new['headline'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
    df_new['summary'] = df_new['summary'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
    df_new.dropna(subset=['symbols', 'headline'], inplace=True)
    
    def contains_target_symbol(symbols_str):
        return any(hedef_sembol in str(symbols_str) for hedef_sembol in HEDEF_SEMBOLLER)
    df_new_filtered = df_new[df_new['symbols'].apply(contains_target_symbol)].copy()

    def create_rag_content(row):
        headline = row['headline']
        summary = row['summary']
        if pd.notna(summary) and len(str(summary).split()) > 5:
            return f"{headline}. {summary}"
        else:
            return headline
    df_new_filtered['rag_content'] = df_new_filtered.apply(create_rag_content, axis=1)
    
    if df_new_filtered.empty:
        print("Filtreleme sonrası işlenecek yeni haber bulunamadı.")
        return
        
    print(f"{len(df_new_filtered)} adet yeni haber işlendi ve hafızaya eklenmeye hazır.")

    # --- Adım 2: Mevcut Vektör Veritabanını Yükle ve Güncelle (DEĞİŞİKLİK YOK) ---
    print("Mevcut vektör veritabanı yükleniyor...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    print("Yeni haberler vektör veritabanına ekleniyor...")
    new_documents = [
        Document(
            page_content=row['rag_content'],
            metadata={'source': row['source'], 'title': row['headline'], 'publish_date': row['timestamp']}
        ) for index, row in df_new_filtered.iterrows()
    ]
    vector_store.add_documents(new_documents)
    print("Vektör veritabanı başarıyla güncellendi!")

    # --- Adım 3: Ana Arşiv CSV Dosyasını Güncelle (BURASI DEĞİŞTİ) ---
    print("Ana CSV arşivi güncelleniyor...")
    final_df_new = df_new_filtered[['timestamp', 'headline', 'rag_content', 'source', 'symbols']].copy()
    final_df_new.rename(columns={'headline': 'title'}, inplace=True)
    
    # --- YENİ VE BASİT SIRALAMA MANTIĞI ---
    # 1. Mevcut knowledge_base.csv'yi oku (varsa).
    try:
        df_existing = pd.read_csv(KNOWLEDGE_BASE_CSV)
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    # 2. Eski ve yeni verileri birleştir.
    df_combined = pd.concat([df_existing, final_df_new], ignore_index=True)
    
    # 3. Tekrar edenleri (aynı başlık ve zaman damgasına sahip olanlar) kaldır.
    df_combined.drop_duplicates(subset=['timestamp', 'title'], inplace=True)
    
    # 4. Tüm birleşmiş veriyi tarihe göre sırala (en yeni en üstte olacak şekilde).
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.sort_values(by='timestamp', ascending=False, inplace=True)
    
    # 5. Sıralanmış bu tam listeyi dosyanın üzerine tamamen yaz.
    df_combined.to_csv(KNOWLEDGE_BASE_CSV, index=False, encoding='utf-8-sig')
    
    print(f"Ana arşiv '{KNOWLEDGE_BASE_CSV}' başarıyla birleştirildi, sıralandı ve kaydedildi.")
    print(f"Arşivdeki toplam haber sayısı: {len(df_combined)}")
    print("\nTüm güncelleme işlemleri tamamlandı!")


if __name__ == '__main__':
    update_knowledge_base()