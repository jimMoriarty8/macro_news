# main_controller.py dosyanızın son hali

import os
import html
import logging
import pandas as pd
import asyncio
from dotenv import load_dotenv
from alpaca.data.live.news import NewsDataStream
from langchain.docstore.document import Document
from langchain_chroma import Chroma

# --- YENİ EKLENEN IMPORTLAR ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- YENİ IMPORTLAR SONU ---

# Kendi dosyalarımızdan importlar
from analysis_engine import (
    initialize_analyst_assistant, 
    parse_analyst_report, 
    send_telegram_message, 
    get_btc_price
)
import config # config.py'yi de import edelim

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR ---
# Bu ayarlar artık config.py'den okunacak ama burada kalabilirler
# veya doğrudan config modülünden çağrılabilirler.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
IMPACT_THRESHOLD = config.IMPACT_THRESHOLD
SYMBOL_WATCHLIST = config.TAKIP_EDILEN_SEMBOLLER

# --- 2. RAG SİSTEMİ KURULUMU ---
# Program başlarken, analiz ve güncelleme için gerekli olan 3 aracı birden alıyoruz.
retriever, document_chain, vector_store = initialize_analyst_assistant()

# --- YENİ EKLENEN BÖLÜM: ÇEVİRİ MOTORU ---
# Sadece çeviri yapmak için hızlı ve verimli bir LLM zinciri oluşturuyoruz.
try:
    translator_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    translator_prompt = ChatPromptTemplate.from_template(
        "Translate the following English financial headline into Turkish. Provide only the Turkish translation, nothing else.\n\nEnglish Headline: {headline}\n\nTurkish Translation:"
    )
    translator_chain = translator_prompt | translator_llm | StrOutputParser()
    print("Çeviri motoru başarıyla yüklendi.")
except Exception as e:
    print(f"UYARI: Çeviri motoru yüklenemedi. Hata: {e}")
    translator_chain = None
# --- ÇEVİRİ MOTORU SONU ---


# --- 3. ARKA PLAN VERİTABANI GÜNCELLEME GÖREVİ ("ŞEF") ---
# Bu fonksiyonunuzda bir değişiklik yapmaya gerek yok.
async def process_and_save_in_background(news_dict: dict, vs: Chroma):
    # ... (Mevcut kodunuz burada kalacak)
    pass # Örnek olarak geçildi


# --- 4. CANLI HABER ANALİZ FONKSİYONU ("GARSON") ---
async def analyze_news_on_arrival(data):
    """
    Hızlıca haberi alır, analiz eder ve yavaş olan kaydetme işini arka plana atar.
    """
    try:
        # İlgililik kontrolü (Bu kısım sizde zaten vardı)
        is_relevant = any(watched_symbol in str(data.symbols) for watched_symbol in SYMBOL_WATCHLIST)
        if not is_relevant:
            return

        headline_en = html.unescape(data.headline)
        print(f"\n📰 [İLGİLİ HABER GELDİ] {headline_en}")

        # --- YENİ EKLENEN BÖLÜM: HABERİ TÜRKÇE'YE ÇEVİRME ---
        headline_tr = ""
        if translator_chain:
            print("   -> Başlık Türkçe'ye çevriliyor...")
            try:
                headline_tr = await translator_chain.ainvoke({"headline": headline_en})
                print(f"   -> Çeviri: {headline_tr}")
            except Exception as e:
                print(f"   -> Çeviri sırasında hata: {e}")
                headline_tr = "(Çeviri yapılamadı)"
        # --- ÇEVİRİ SONU ---

        # Hızlı Analiz ve Alarm Kısmı
        print("   -> Analiz ediliyor...")
        # Not: Sizin kodunuzda retriever ve document_chain'i doğrudan kullandığınızı varsayıyorum.
        # Bu satırları kendi kodunuzdaki invoke satırıyla eşleştirin.
        retrieved_docs = retriever.get_relevant_documents(headline_en)
        retrieved_docs.sort(key=lambda x: x.metadata.get('publish_date', '1970-01-01'), reverse=True)
        report_text = document_chain.invoke({"input": headline_en, "context": retrieved_docs})
        
        print("\n--- ANALYST REPORT ---")
        print(report_text)
        
        parsed_report = parse_analyst_report(report_text)
        if parsed_report and parsed_report.get('confidence', 0) >= CONFIDENCE_THRESHOLD and parsed_report.get('impact', 0) >= IMPACT_THRESHOLD and parsed_report.get('direction').lower() != 'neutral':
            print(f"✅ ALARM KRİTERLERİ KARŞILANDI!")
            btc_price = get_btc_price()
            direction = parsed_report.get('direction', 'N/A')
            direction_emoji = "🟢" if direction.lower() == 'positive' else "🔴" if direction.lower() == 'negative' else "⚪️"
            
            # --- YENİ EKLENEN BÖLÜM: TELEGRAM MESAJINI GÜNCELLEME ---
            message = (
                f"{direction_emoji} *High-Potential Signal: {direction.upper()}*\n"
                f"*BTC/USDT Price:* `{btc_price}`\n\n"
                f"*Haber (TR):*\n`{headline_tr}`\n\n"
                f"*Headline (EN):*\n`{headline_en}`\n\n"
                f"*Scores:*\n"
                f"Impact: *{parsed_report.get('impact')}/10* | Confidence: *{parsed_report.get('confidence')}/10*\n\n"
                f"*Analyst Comment:*\n_{parsed_report.get('analysis', '')}_"
            )
            # --- MESAJ GÜNCELLEME SONU ---
            send_telegram_message(message)
        else:
            print("❌ Alarm kriterleri karşılanmadı veya yön 'Neutral'.")
        
        # --- YAVAŞ İŞİ ARKA PLANA ATMA ---
        # Bu kısım sizde zaten vardı
        news_item_dict = {"id": data.id, "timestamp": data.created_at, "headline": data.headline, "summary": data.summary, "source": data.source, "symbols": ",".join(data.symbols) if data.symbols else ""}
        asyncio.create_task(process_and_save_in_background(news_item_dict, vector_store))

    except Exception as e:
        print(f"\n🚨 ANA ANALİZ DÖNGÜSÜ HATASI: {e}")


# --- 5. ANA UYGULAMAYI BAŞLATMA ---
# Bu kısım sizde zaten vardı
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    news_stream.subscribe_news(analyze_news_on_arrival, '*')
    print(f"--- CANLI HABER ANALİZ SİSTEMİ AKTİF (ANLIK ÖĞRENME MODU) ---")
    print(f"İzleme Listesi: {list(SYMBOL_WATCHLIST)}")
    news_stream.run()
