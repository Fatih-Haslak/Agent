import requests
import re
from typing import Dict, Optional
from langchain_core.tools import tool


WIKI_API = "https://tr.wikipedia.org/w/api.php"
TIMEOUT = 15
HEADERS = {"User-Agent": "MultiAgentSystem/2.0 (LangGraph; Turkish NLP)"}
WIKI_SUMMARY_MAX = 2000


@tool
def wiki_search(query: str, max_chars: int = 1500) -> str:
    """Türkçe Wikipedia'da arama yapar ve içerik çeker.
    
    Args:
        query: Arama sorgusu (örn: 'Atatürk', 'Sergen Yalçın')
        max_chars: Döndürülecek maksimum karakter sayısı
    """
    try:
        # 1. Arama terimini temizle (kimdir/nedir gibi ekleri çıkar)
        cleaned = re.sub(r"\b(kimdir|nedir|hakkında|ne zaman|nerede|nasıl|kaç)\b",
                         "", query, flags=re.IGNORECASE).strip() or query
        
        # 2. Wikipedia'da başlık bul
        title = _find_title(cleaned)
        if not title:
            return f"Wikipedia'da '{query}' için sonuç bulunamadı."
        
        # 3. İçerik çek
        result = _fetch_page(title, max_chars)
        if not result:
            return f"'{title}' sayfası bulundu ancak içerik çekilemedi."
        
        return (
            f"📚 Wikipedia: {result['title']}\n"
            f"🔗 {result['url']}\n"
            f"📏 {result['char_count']} karakter\n\n"
            f"{result['summary']}"
        )
    except Exception as e:
        return f"Wikipedia hatası: {str(e)}"


def _find_title(query: str) -> Optional[str]:
    """Wikipedia arama API ile başlık bulur."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    r = requests.get(WIKI_API, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    results = r.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def _fetch_page(title: str, max_chars: int) -> Optional[Dict[str, str]]:
    """Wikipedia sayfasından içerik çeker."""
    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json",
        "inprop": "url"
    }
    r = requests.get(WIKI_API, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    page = list(pages.values())[0]
    
    extract = page.get("extract", "")
    if not extract:
        return None
    
    summary = extract[:max_chars]
    if len(extract) > max_chars:
        summary += "…"
    
    return {
        "title": title,
        "summary": summary,
        "url": page.get("fullurl", f"https://tr.wikipedia.org/wiki/{title.replace(' ', '_')}"),
        "char_count": str(len(extract))
    }
