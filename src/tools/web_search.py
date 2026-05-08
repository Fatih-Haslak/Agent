import json
from typing import List
from langchain_core.tools import tool
from ddgs import DDGS


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Web'de arama yapar ve sonuçları JSON formatında döndürür.
    
    Args:
        query: Arama sorgusu
        max_results: Döndürülecek maksimum sonuç sayısı (varsayılan: 3)
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            if not results:
                return "Arama sonucu bulunamadı."
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")[:500]
                })
            return json.dumps(formatted, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Arama sırasında hata oluştu: {str(e)}"


@tool
def summarize(text: str, max_words: int = 100) -> str:
    """Verilen metni belirtilen kelime sınırına kadar özetler.
    
    Args:
        text: Özetlenecek metin
        max_words: Maksimum kelime sayısı (varsayılan: 100)
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    # Basit bir özet stratejisi: ilk ve son cümleleri birleştirerek kırp
    summary = " ".join(words[:max_words]) + "..."
    return summary
