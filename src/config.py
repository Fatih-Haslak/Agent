"""
LLM Yapılandırması — Local Turkish-Gemma-9b Model
Enhanced with Streaming + ReAct Pattern Support
"""

import os
import re
import json
import torch
from typing import List, Dict, Any, Optional, Generator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread

MODEL_ID = "ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"
USE_4BIT = os.getenv("USE_4BIT", "true").lower() in ("true", "1", "yes")


class ModelLoader:
    """HuggingFace modelini 4-bit quantization ile yükler."""

    _tokenizer = None
    _model = None

    @classmethod
    def load(cls, model_id: str = MODEL_ID, use_4bit: bool = USE_4BIT):
        if cls._tokenizer is not None and cls._model is not None:
            return cls._tokenizer, cls._model

        print(f"📥 Model indiriliyor/yükleniyor: {model_id}")
        print(f"   CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)

        if use_4bit:
            print("   Mod: 4-bit NF4 Quantization")
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=qcfg,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        else:
            print("   Mod: float16")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=True
            )

        model.eval()
        cls._tokenizer = tokenizer
        cls._model = model
        print("✅ Model hazır!")
        return tokenizer, model


class LLMEngine:
    """Dil modeli çıkarım motoru - Streaming destekli."""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def _terminators(self):
        terms = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            try:
                eot = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if isinstance(eot, int) and eot != self.tokenizer.unk_token_id:
                    terms.append(eot)
            except Exception:
                pass
        return terms

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Mesajları prompt formatına dönüştürür."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = ""
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    prompt += f"Sistem: {content}\n"
                elif role == "user":
                    prompt += f"Kullanıcı: {content}\n"
                else:
                    prompt += f"Asistan: {content}\n"
            prompt += "Asistan:"
            return prompt

    @torch.inference_mode()
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=self._terminators(),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def generate_stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 256, temperature: float = 0.7) -> Generator[str, None, None]:
        """Token-by-token streaming üretir."""
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95,
            repetition_penalty=1.05,
            eos_token_id=self._terminators(),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            if text:
                yield text
        
        thread.join()

    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.generate(messages, max_new_tokens, temperature)

    def chat_stream(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> Generator[str, None, None]:
        """Streaming chat - token by token."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        yield from self.generate_stream(messages, max_new_tokens, temperature)

    def chat_with_history(self, system_prompt: str, user_prompt: str, history, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    role = getattr(msg, 'type', 'user')
                    if role == 'human':
                        role = 'user'
                    elif role == 'ai':
                        role = 'assistant'
                    content = getattr(msg, 'content', str(msg))
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        return self.generate(messages, max_new_tokens, temperature)

    def chat_with_history_stream(self, system_prompt: str, user_prompt: str, history, max_new_tokens: int = 256, temperature: float = 0.7) -> Generator[str, None, None]:
        """Streaming chat with history."""
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    messages.append(msg)
                else:
                    role = getattr(msg, 'type', 'user')
                    if role == 'human':
                        role = 'user'
                    elif role == 'ai':
                        role = 'assistant'
                    content = getattr(msg, 'content', str(msg))
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        yield from self.generate_stream(messages, max_new_tokens, temperature)


# Singleton instance
_llm_engine = None

def get_llm_engine() -> LLMEngine:
    """Global LLM engine instance döndürür (lazy load)."""
    global _llm_engine
    if _llm_engine is None:
        tokenizer, model = ModelLoader.load()
        _llm_engine = LLMEngine(tokenizer, model)
    return _llm_engine


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Metin içinden ilk JSON objesini çıkarır."""
    if not text:
        return None
    
    # 1. JSON kod blokları
    code_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 1b. Inline ```json {...} ```
    inline_code = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if inline_code:
        try:
            return json.loads(inline_code.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 2. Stack-based süslü parantez eşleştirme
    best_match = None
    best_depth = 0
    
    for start in range(len(text)):
        if text[start] == '{':
            depth = 1
            for end in range(start + 1, len(text)):
                if text[end] == '{':
                    depth += 1
                elif text[end] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:end+1]
                        try:
                            parsed = json.loads(candidate)
                            if candidate.count('{') > best_depth:
                                best_match = parsed
                                best_depth = candidate.count('{')
                        except json.JSONDecodeError:
                            pass
                        break
    
    if best_match is not None:
        return best_match
    
    # 3. Basit regex fallback
    match = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None
