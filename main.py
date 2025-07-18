# main_controller.py (Düzeltilmiş ve Sağlamlaştırılmış Versiyon)

import os
import html
import logging
import asyncio
from dotenv import load_dotenv
from alpaca.data.live.news import NewsDataStream
from langchain.docstore.document import Document
from langchain_chroma import Chroma

# Gerekli LangChain importları
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Kendi dosyalarımızdan importlar
from analysis_engine import (
    initialize_analyst_assistant, 
    parse_analyst_report, 
    send_telegram_message, 
    get_btc_price
)
import config

# .env dosyasını yükle
load_dotenv()

# --- 1. AYARLAR ---
# Ayarları config.py dosyasından okuyoruz
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
IMPACT_THRESHOLD = config.IMPACT_THRESHOLD
SYMBOL_WATCHLIST = config.SYMBOL_WATCHLIST


# --- 2. RAG SİSTEMİ KURULUMU ---
# Program başlarken, app.py'den sadece temel araçları alıyoruz
retriever, _, vector_store = initialize_analyst_assistant()

# --- YENİ VE DAHA SAĞLAM ZİNCİR YAPISI (LCEL) ---
# LLM ve Prompt'u doğrudan bu dosyada, config'den gelen ayarlarla tanımlıyoruz.
# Bu, önceki AttributeError hatasını çözer.
llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL, temperature=0.2)
prompt = ChatPromptTemplate.from_template(config.SYSTEM_PROMPT)

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# --- ZİNCİR YAPISI SONU ---

# --- ÇEVİRİ MOTORU ---
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


# --- 3. ARKA PLAN GÖREVİ ---
async def process_and_save_in_background(news_dict: dict, vs: Chroma):
    # ... (Bu kısım sizin kodunuzdaki gibi kalabilir)
    pass 


# --- 4. CANLI HABER ANALİZ FONKSİYONU ---
async def analyze_news_on_arrival(data):
    """
    Haberi alır, analiz eder ve kaydetme işini arka plana atar.
    """
    try:
        is_relevant = any(watched_symbol in str(data.symbols) for watched_symbol in SYMBOL_WATCHLIST)
        if not is_relevant:
            return

        headline_en = html.unescape(data.headline)
        print(f"\n📰 [İLGİLİ HABER GELDİ] {headline_en}")

        # Çeviri adımı
        headline_tr = ""
        if translator_chain:
            print("   -> Başlık Türkçe'ye çevriliyor...")
            try:
                headline_tr = await translator_chain.ainvoke({"headline": headline_en})
                print(f"   -> Çeviri başarılı: {headline_tr}")
            except Exception as e:
                headline_tr = f"(Çeviri hatası: {e})"
        
        # Analiz adımı
        print("   -> Analiz ediliyor...")
        report_text = await rag_chain.ainvoke(headline_en)
        
        print("\n--- ANALYST REPORT ---")
        print(report_text)
        
        parsed_report = parse_analyst_report(report_text)
        if parsed_report and parsed_report.get('confidence', 0) >= CONFIDENCE_THRESHOLD and parsed_report.get('impact', 0) >= IMPACT_THRESHOLD and parsed_report.get('direction').lower() != 'neutral':
            print(f"✅ ALARM KRİTERLERİ KARŞILANDI!")
            btc_price = get_btc_price()
            direction = parsed_report.get('direction', 'N/A')
            direction_emoji = "🟢" if direction.lower() == 'positive' else "🔴"
            
            message = (
                f"{direction_emoji} *Signal: {direction.upper()}*\n"
                f"*BTC/USDT Price:* `{btc_price}`\n\n"
                f"*Haber (TR):*\n`{headline_tr}`\n\n"
                f"*Headline (EN):*\n`{headline_en}`\n\n"
                f"*Scores:*\n"
                f"Impact: *{parsed_report.get('impact')}/10* | Confidence: *{parsed_report.get('confidence')}/10*\n\n"
                f"*Commentary:*\n_{parsed_report.get('analysis', '')}_"
            )
            send_telegram_message(message)
        else:
            print("❌ Alarm kriterleri karşılanmadı veya yön 'Neutral'.")
        
        news_item_dict = {"id": data.id, "timestamp": data.created_at, "headline": data.headline, "summary": data.summary, "source": data.source, "symbols": ",".join(data.symbols) if data.symbols else ""}
        asyncio.create_task(process_and_save_in_background(news_item_dict, vector_store))

    except Exception as e:
        print(f"\n🚨 ANA ANALİZ DÖNGÜSÜ HATASI: {e}")


# --- 5. ANA UYGULAMAYI BAŞLATMA ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    news_stream = NewsDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    news_stream.subscribe_news(analyze_news_on_arrival, '*')
    print(f"--- CANLI HABER ANALİZ SİSTEMİ AKTİF ---")
    print(f"İzleme Listesi: {list(SYMBOL_WATCHLIST)}")
    news_stream.run()
