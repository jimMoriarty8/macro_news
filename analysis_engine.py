# analysis_engine.py

import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import config

def initialize_analyst_assistant():
    """
    Vektör veritabanını, LLM'i ve temel RAG parçalarını kurar.
    main.py'nin hem analiz hem de veritabanı güncellemesi yapabilmesi için
    gerekli olan 3 aracı (retriever, document_chain, vector_store) döndürür.
    """
    print("RAG Analist Asistanı başlatılıyor...")
    # API anahtarını doğrudan ortam değişkenlerinden al
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Hata mesajını aranan anahtarla tutarlı hale getiriyoruz.
        raise ValueError("HATA: GOOGLE_API_KEY ortam değişkeni bulunamadı! Lütfen Render'daki Environment Group'u kontrol edin.")

    # DÜZELTME 2: Anahtarı DOĞRUDAN modellere parametre olarak iletiyoruz.
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    
    # Veritabanı var mı diye kontrol et, yoksa boş oluştur.
    if os.path.exists(config.CHROMA_DB_PATH):
        print(f"Mevcut vektör veritabanı '{config.CHROMA_DB_PATH}' klasöründen yükleniyor...")
        vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
        print("Veritabanı başarıyla yüklendi.")
    else:
        print("UYARI: Henüz bir veritabanı bulunamadı.")
        print("Sıfırdan BOŞ bir vektör veritabanı oluşturuluyor...")
        # Kütüphanenin hata vermemesi için içeriği boş olan geçici bir dökümanla başlatıyoruz
        placeholder_doc = Document(page_content="initialization_document")
        vector_store = Chroma.from_documents(
            documents=[placeholder_doc], 
            embedding=embeddings, 
            persist_directory=config.CHROMA_DB_PATH
        )
        print(f"Boş veritabanı '{config.CHROMA_DB_PATH}' klasöründe başarıyla oluşturuldu.")

    # LangChain araçlarını oluştur
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    
    prompt = ChatPromptTemplate.from_template('''You are a world-class financial analyst, specializing in both macroeconomic trends and specific crypto catalysts. Your task is to analyze the 'USER INPUT' (a new headline) by first categorizing it and then applying the appropriate analytical framework based on the provided 'CONTEXT'.

---
**ANALYTICAL FRAMEWORKS (Kural Setleri)**

**ÇERÇEVE 1: Makroekonomik Rejim (FED Odaklı)**
**KULLANIM KURALI:** Bu çerçeveyi **SADECE VE SADECE** resmi kurumlar (örn: ABD Çalışma İstatistikleri Bürosu, Merkez Bankaları) tarafından yayınlanan ve piyasanın genelini etkileyen ekonomik takvim verileri için kullan. (Örnekler: TÜFE/CPI, ÜFE/PPI, Tarım Dışı İstihdam/NFP, GSYİH/GDP, Perakende Satışlar).
- **Kural 1.1 (Kötü Haber İyidir):** Zayıflayan bir ekonomiyi gösteren veriler, FED'in faiz indirme olasılığını artırdığı için kripto için **POZİTİF** kabul edilir.
- **Kural 1.2 (İyi Haber Kötüdür):** Güçlü bir ekonomiyi gösteren veriler, "faizlerin daha uzun süre yüksek kalacağı" politikasını güçlendirdiği için kripto için **NEGATİF** kabul edilir.

**ÇERÇEVE 2: Katalizör Odaklı Rejim (Spesifik Olaylar)**
**KULLANIM KURALI:** Bu çerçeveyi, haberde **belirli bir coin, şirket, kişi veya spesifik bir olaydan** bahsediliyorsa kullan.
- **Kural 2.1 (Hype ve Duygu):** Elon Musk, Donald Trump gibi küresel çapta tanınan ve piyasayı etkileme gücü olan bir figürün kriptoyu güçlü bir şekilde desteklemesi, anlık etki potansiyeli yüksek, çok güçlü bir **POZİTİF** sinyaldir.
- **Kural 2.2 (Kabul ve Teknoloji):** Büyük bir şirketin bir coini kullanacağını duyurması, büyük bir borsa listelemesi (Coinbase, Binance) veya önemli bir teknolojik güncellemenin başarıyla tamamlanması, ilgili varlık ve genel piyasa için **POZİTİF** kabul edilir.
- **Kural 2.3 (Regülasyon):** SEC'in büyük bir projeye dava açması gibi haberler **NEGATİF** kabul edilir. Olumlu regülasyon haberleri ise **POZİTİF**'tir.
- **KURAL 2.4 (Dolaylı Etki / Sektörel Duygu):** Microsoft, Nvidia gibi büyük teknoloji şirketleri veya genel finans piyasaları hakkındaki olumlu/olumsuz haberler, kripto üzerinde doğrudan bir etki yaratmaz. Bu tür haberleri, teknoloji/risk iştahı üzerindeki dolaylı **duygu sinyali** olarak yorumla ve 'Impact Score'unu **düşük (1-3 aralığında)** tut.

---
**GÖREVİN (Adım Adım):**

1.  **Kategorize Et (EN ÖNEMLİ ADIM):** İlk olarak 'USER INPUT' başlığını analiz et. **Eşitliği Bozma Kuralı: Eğer haberde spesifik bir varlık, şirket veya kişi adı geçiyorsa (örn: 'Bitcoin', 'Microsoft', 'Trump'), her zaman öncelikli olarak ÇERÇEVE 2'yi (Katalizör Odaklı) kullanmak ZORUNDASIN.** Çerçeve 1'i sadece haber, spesifik bir varlıktan çok piyasanın geneline yönelik resmi bir ekonomik veri ise kullanmalısın.
2.  **Analiz Et:** Seçtiğin çerçevenin kurallarını uygula.
3.  **Puanla:** Impact ve Confidence skorlarını ata. Somut bir eylem (örn: "Coinbase, X coinini listeledi") sadece bir fikirden ("Trump, Bitcoin'i seviyor") daha yüksek bir 'Confidence Score'a sahip olmalıdır. Haberin kaynağının gücü 'Impact Score'u etkiler.
4.  **Raporla:** Aşağıdaki yapılandırılmış raporu doldur. 'Analysis' bölümünde, hangi çerçeveyi kullandığını ve nedenini **açıkça belirt.**

---
CONTEXT (Historical Precedents):
{context}

USER INPUT (New, Breaking Headline):
{input}
---

**STRUCTURED ANALYSIS REPORT:**
**Direction:** [Positive, Negative, Neutral]
**Impact Score:** [1-10]
**Confidence Score:** [1-10]
**Analysis:** [A concise, 1-2 sentence reasoning. Explicitly state WHICH framework (Macroeconomic or Catalyst-Driven) you used and why the news fits its rules.]''')
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    print("\nAnalist Asistanı'nın tüm parçaları hazır.")
    print("="*50)
    
    # main.py'nin ihtiyaç duyduğu 3 temel aracı döndür
    return retriever, document_chain, vector_store

# --- Diğer Yardımcı Fonksiyonlar ---
# Projenin diğer dosyalarında (main.py gibi) kullanılacak olan
# parse_analyst_report, send_telegram_message ve get_btc_price
# fonksiyonlarını da bu "motor" dosyasında merkezi olarak tutmak iyi bir fikirdir.

import re
import requests
from binance.client import Client as BinanceClient

def get_btc_price():
    """Anlık BTC/USDT fiyatını çeker."""
    try:
        client = BinanceClient()
        ticker = client.get_symbol_ticker(symbol="BTCUSDT")
        price = float(ticker['price'])
        return f"${price:,.2f}"
    except Exception as e:
        print(f"UYARI: Anlık BTC fiyatı çekilemedi. Hata: {e}")
        return "Price N/A"

def send_telegram_message(message):
    """Belirtilen mesajı Telegram'a gönderir."""
    TOKEN = os.getenv(config.TELEGRAM_BOT_TOKEN_ENV)
    CHAT_ID = os.getenv(config.TELEGRAM_CHAT_ID_ENV)
    
    if not TOKEN or not CHAT_ID:
        print("\n>> UYARI: Telegram bilgileri eksik. Mesaj gönderilemedi.")
        return
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("\n>>> ALARM TELEGRAM'A BAŞARIYLA GÖNDERİLDİ <<<")
        else:
            print(f"\n>> HATA: Telegram'a mesaj gönderilemedi. Durum Kodu: {response.status_code}, Cevap: {response.text}")
    except Exception as e:
        print(f"\n>> HATA: Telegram mesajı gönderilirken bir sorun oluştu: {e}")

def parse_analyst_report(report_text):
    """LLM'den gelen metin raporunu ayrıştırır."""
    try:
        direction = re.search(r"\*\*?Direction:\*\*?\s*\[?([^\]\n]+)\]?", report_text, re.IGNORECASE).group(1).strip()
        impact = int(re.search(r"\*\*?Impact Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        confidence = int(re.search(r"\*\*?Confidence Score:\*\*?\s*\[?(\d+)", report_text, re.IGNORECASE).group(1))
        analysis_match = re.search(r"\*\*?Analysis:\*\*?\s*([\s\S]*)", report_text, re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else "Analiz metni bulunamadı."
        return {"direction": direction, "impact": impact, "confidence": confidence, "analysis": analysis}
    except (AttributeError, IndexError, ValueError) as e:
        print(f"AYRIŞTIRMA HATASI: Rapor beklenen formatta değil. Hata: {e}")
        return None