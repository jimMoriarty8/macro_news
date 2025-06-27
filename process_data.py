import pandas as pd
import os
import html

# --- AYARLAR (DEĞİŞİKLİK YOK) ---
INPUT_CSV = "raw_news_archive.csv" 
OUTPUT_CSV = "knowledge_base.csv"
HEDEF_SEMBOLLER = [
    'BTC/USD', "BTC", 'BTCUSD',
    'ETH/USD', "ETH", 'ETHUSD',
    "SOL/USD", "SOL", 'SOLUSD',
    "XRP/USD", "XRP", 'XRPUSD',
    "BNB/USD", "BNB", 'BNBUSD',
    'SPY', 'QQQ'
]

print("="*50)
print("VERİ İŞLEME TEŞHİS MODU BAŞLATILDI")
print("="*50)

# --- Adım 1: Ham Veriyi Yükle ---
try:
    df = pd.read_csv(INPUT_CSV)
    print(f"[TEŞHİS] Adım 1: '{INPUT_CSV}' dosyasından {len(df)} adet ham haber yüklendi.")
except FileNotFoundError:
    print(f"HATA: '{INPUT_CSV}' dosyası bulunamadı.")
    exit()

# --- Adım 2: Eksik Verileri Temizle ---
df_cleaned = df.dropna(subset=['symbols', 'headline', 'id']).copy()
kayip_veri_sayisi = len(df) - len(df_cleaned)
print(f"[TEŞHİS] Adım 2: Eksik ('id', 'headline', 'symbols') verisi olan {kayip_veri_sayisi} satır silindi.")
print(f"           -> Kalan haber sayısı: {len(df_cleaned)}")

# --- Adım 3: Sembol Filtrelemesi ---
def iceriyor_mu(symbols_str):
    # Gelen sembol listesinin (string formatında) hedef sembollerimizden herhangi birini içerip içermediğini kontrol et
    return any(hedef_sembol in str(symbols_str) for hedef_sembol in HEDEF_SEMBOLLER)

# Filtrelemeden ÖNCE ve SONRA kaç haber olduğunu görelim
df_filtrelenmis = df_cleaned[df_cleaned['symbols'].apply(iceriyor_mu)].copy()
filtrelenen_veri_sayisi = len(df_cleaned) - len(df_filtrelenmis)
print(f"[TEŞHİS] Adım 3: Hedef sembolleri içermeyen {filtrelenen_veri_sayisi} satır filtrelendi.")
print(f"           -> Kalan haber sayısı: {len(df_filtrelenmis)}")

# --- Eğer hiç haber kalmadıysa, sorunu belirt ve dur ---
if len(df_filtrelenmis) == 0:
    print("\n[HATA] Filtreleme sonrası hiç haber kalmadı!")
    print("Lütfen 'raw_news_archive.csv' dosyasındaki 'symbols' sütununun içeriğini ve HEDEF_SEMBOLLER listesini kontrol edin.")
    exit()

# --- Adım 4: HTML Temizliği ve İçerik Oluşturma (Bu adımlarda sorun beklemiyoruz) ---
df_filtrelenmis['headline'] = df_filtrelenmis['headline'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)
df_filtrelenmis['summary'] = df_filtrelenmis['summary'].apply(lambda x: html.unescape(x) if isinstance(x, str) else x)

def create_rag_content(row):
    headline = row['headline']
    summary = row['summary']
    if pd.notna(summary) and len(str(summary).split()) > 5:
        return f"{headline}. {summary}"
    else:
        return headline
df_filtrelenmis['rag_content'] = df_filtrelenmis.apply(create_rag_content, axis=1)

# --- Adım 5: Nihai Dosyayı Kaydet ---
final_df = df_filtrelenmis[['id', 'timestamp', 'headline', 'rag_content', 'source', 'symbols']].copy()
final_df.rename(columns={'headline': 'title'}, inplace=True)
final_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print("\n" + "="*50)
print("TEŞHİS TAMAMLANDI")
print(f"'{OUTPUT_CSV}' dosyasına {len(final_df)} adet haber başarıyla yazıldı.")
print("="*50)