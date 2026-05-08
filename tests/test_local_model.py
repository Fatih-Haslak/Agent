import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"

def test_model():
    print(f"Model yükleniyor: {MODEL_ID}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print("Model 4-bit quantization ile yükleniyor (bu biraz zaman alabilir)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    print("Pipeline oluşturuluyor...")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.1
    )
    
    # Test 1: Basit Türkçe soru
    print("\n--- Test 1: Basit Türkçe Soru ---")
    prompt1 = "Soru: Türkiye'nin başkenti neresidir?\nCevap:"
    result1 = pipe(prompt1)[0]["generated_text"]
    print(f"Prompt: {prompt1}")
    print(f"Sonuç: {result1}")
    
    # Test 2: Tool calling formatı
    print("\n--- Test 2: Tool Calling Formatı ---")
    prompt2 = """Sen bir yapay zeka asistanısın. Aşağıdaki araçları kullanabilirsin:
- calculator: Matematiksel hesaplama yapar
- web_search: Web'de arama yapar

Kullanıcı: 15 çarpı 23 kaç eder?
Düşünce:"""
    result2 = pipe(prompt2)[0]["generated_text"]
    print(f"Prompt: {prompt2}")
    print(f"Sonuç: {result2}")
    
    # Test 3: LangChain HuggingFacePipeline
    print("\n--- Test 3: LangChain Entegrasyonu ---")
    llm = HuggingFacePipeline(pipeline=pipe)
    result3 = llm.invoke("Türkiye'deki en büyük şehir hangisidir?")
    print(f"LangChain Sonuç: {result3}")
    
    print("\n✅ Model testi tamamlandı!")
    return llm

if __name__ == "__main__":
    test_model()
