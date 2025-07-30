# ask_analyst.py

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import config
from analysis_engine import initialize_analyst_components

def main():
    """
    Kullanıcının veritabanıyla sohbet etmesini sağlayan interaktif bir komut satırı arayüzü.
    """
    load_dotenv()
    
    # Temel RAG bileşenlerini yüklüyoruz (LLM ve veritabanı retriever'ı)
    print("Analistin beyni (vektör veritabanı) yükleniyor... Lütfen bekleyin.")
    llm, retriever, _ = initialize_analyst_components()

    # Bu betiğe özel sohbet zincirini oluşturuyoruz.
    chat_prompt = ChatPromptTemplate.from_template(config.CHAT_PROMPT)
    chat_chain = create_stuff_documents_chain(llm, chat_prompt)

    print("\n✅ Analist hazır. Sorularınızı sorabilirsiniz.")
    print('Sohbeti bitirmek için "exit" veya "çıkış" yazın.')

    while True:
        question = input("\n> ")
        if question.lower() in ["exit", "quit", "çıkış"]:
            print("Görüşmek üzere!")
            break
        
        # --- GÜVENLİK KONTROLÜ ---
        # Eğer input boş veya sadece boşluk karakterlerinden oluşuyorsa,
        # işleme almadan döngünün başına dön ve tekrar sor.
        if not question.strip():
            continue
        
        print("Analist düşünüyor...")
        # Kullanıcının sorusuna en uygun belgeleri veritabanından bul ve zinciri çalıştır
        context_docs = retriever.invoke(question)
        response = chat_chain.invoke({"input": question, "context": context_docs})
        
        print("\n--- Analistin Cevabı ---")
        print(response)
        print("-" * 24)

if __name__ == "__main__":
    main()
