import os
import json
import subprocess
import tempfile
from typing import Optional
from langchain_core.tools import tool


@tool
def code_exec(language: str, code: str, timeout: int = 10) -> str:
    """Verilen kodu güvenli bir şekilde çalıştırır ve çıktısını döndürür.
    Kritik bir tool'dur; human-in-the-loop onay gerektirir.
    
    Args:
        language: Programlama dili (python | bash | javascript)
        code: Çalıştırılacak kod
        timeout: Maksimum çalışma süresi (saniye, varsayılan: 10)
    """
    if language.lower() == "python":
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[HATA]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[EXIT CODE]: {result.returncode}"
            return output or "Kod çalıştırıldı, çıktı üretmedi."
        except subprocess.TimeoutExpired:
            return f"Kod {timeout} saniye içinde tamamlanamadı."
        except Exception as e:
            return f"Çalıştırma hatası: {str(e)}"
    
    elif language.lower() in ("bash", "shell"):
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[HATA]: {result.stderr}"
            return output or "Komut çalıştırıldı, çıktı üretmedi."
        except subprocess.TimeoutExpired:
            return f"Komut {timeout} saniye içinde tamamlanamadı."
        except Exception as e:
            return f"Çalıştırma hatası: {str(e)}"
    
    elif language.lower() == "javascript":
        try:
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[HATA]: {result.stderr}"
            return output or "Kod çalıştırıldı, çıktı üretmedi."
        except FileNotFoundError:
            return "Node.js kurulu değil. JavaScript kodu çalıştırılamadı."
        except Exception as e:
            return f"Çalıştırma hatası: {str(e)}"
    
    else:
        return f"Desteklenmeyen dil: {language}. Desteklenenler: python, bash, javascript."


@tool
def file_io(action: str, file_path: str, content: Optional[str] = None) -> str:
    """Dosya okuma/yazma/silme işlemleri gerçekleştirir.
    Yazma ve silme işlemleri kritiktir; human-in-the-loop onay gerektirir.
    
    Args:
        action: İşlem türü (read | write | delete | append)
        file_path: Dosya yolu
        content: Yazılacak/eklenecek içerik (read/delete için opsiyonel)
    """
    # Güvenlik: çalışma dizini dışına çıkma
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    abs_path = os.path.abspath(file_path)
    if not abs_path.startswith(base_dir):
        return "Hata: Çalışma dizini dışındaki dosyalara erişim reddedildi."
    
    try:
        if action == "read":
            if not os.path.exists(abs_path):
                return f"Dosya bulunamadı: {file_path}"
            with open(abs_path, "r", encoding="utf-8") as f:
                return f.read()
        
        elif action == "write":
            if content is None:
                return "Hata: 'write' işlemi için content gerekli."
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Dosya yazıldı: {file_path}"
        
        elif action == "append":
            if content is None:
                return "Hata: 'append' işlemi için content gerekli."
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "a", encoding="utf-8") as f:
                f.write(content)
            return f"Dosyaya eklendi: {file_path}"
        
        elif action == "delete":
            if not os.path.exists(abs_path):
                return f"Dosya bulunamadı: {file_path}"
            os.remove(abs_path)
            return f"Dosya silindi: {file_path}"
        
        else:
            return f"Bilinmeyen işlem: {action}. Desteklenenler: read, write, delete, append."
    
    except Exception as e:
        return f"Dosya işlemi hatası: {str(e)}"
