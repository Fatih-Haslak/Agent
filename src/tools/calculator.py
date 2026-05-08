import json
from typing import Optional
import httpx
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Matematiksel ifadeleri güvenli bir şekilde hesaplar.
    
    Args:
        expression: Hesaplanacak matematiksel ifade (örn: '2 + 2 * 5', 'sqrt(16)')
    """
    try:
        # Güvenlik: sadece temel matematiksel fonksiyonlara izin ver
        allowed_names = {
            "abs": abs,
            "round": round,
            "max": max,
            "min": min,
            "sum": sum,
            "pow": pow,
        }
        code = compile(expression, "<string>", "eval")
        # Sadece saf ifadelere izin ver
        for name in code.co_names:
            if name not in allowed_names:
                return f"Hata: '{name}' fonksiyonuna izin verilmiyor. Sadece temel matematiksel işlemler desteklenir."
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return f"Sonuç: {result}"
    except Exception as e:
        return f"Hesaplama hatası: {str(e)}"


@tool
def http_request(method: str, url: str, headers: Optional[str] = None, body: Optional[str] = None, timeout: int = 10) -> str:
    """HTTP isteği gönderir ve yanıtı döndürür.
    Kritik bir tool'dur; human-in-the-loop onay gerektirebilir.
    
    Args:
        method: HTTP metodu (GET | POST | PUT | DELETE | PATCH)
        url: İstek URL'si
        headers: JSON formatında header'lar (opsiyonel)
        body: İstek gövdesi (opsiyonel)
        timeout: Zaman aşımı (saniye, varsayılan: 10)
    """
    try:
        parsed_headers = {}
        if headers:
            parsed_headers = json.loads(headers)
        
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=parsed_headers,
                content=body.encode("utf-8") if body else None
            )
            output = f"Status: {response.status_code}\n"
            output += f"Headers: {dict(response.headers)}\n"
            # Yanıt boyutunu sınırla
            text = response.text[:2000]
            output += f"Body:\n{text}"
            if len(response.text) > 2000:
                output += "\n... (truncated)"
            return output
    except json.JSONDecodeError:
        return "Hata: Header'lar geçerli JSON formatında değil."
    except Exception as e:
        return f"HTTP isteği hatası: {str(e)}"
