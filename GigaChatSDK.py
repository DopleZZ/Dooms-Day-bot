import os
from typing import List, Tuple
from chunks import retrieve
from env_loader import load_dotenv

load_dotenv()

try:
    from gigachat import GigaChat
except Exception:
    GigaChat = None

GC_TOKEN = os.getenv("GC_TOKEN") or os.getenv("GIGACHAT_CREDENTIALS") or ""

SYSTEM = "Отвечай кратко и строго по контексту на русском. Если данных не хватает — скажи об этом."


def build_prompt(question: str, ctx_items: List[Tuple[str, float]]) -> str:
    ctx_text = "\n\n".join(f"Фрагмент {i+1}:\n{c[0]}" for i, c in enumerate(ctx_items))
    return f"""{SYSTEM}

Вопрос: {question}

Контекст:
{ctx_text}

Ответ:
"""


def rag_answer(question: str):
    ctx = retrieve(question, k=3)
    prompt = build_prompt(question, ctx)
    if not GC_TOKEN:
        return ("GigaChat token не настроен (GC_TOKEN).", ctx)
    if GigaChat is None:
        return ("GigaChat SDK не установлен (import failed). Установите пакет `gigachat` или запустите с mock.", ctx)
    try:
        with GigaChat(credentials=GC_TOKEN, scope="GIGACHAT_API_PERS", verify_ssl_certs=False) as gc:
            resp = gc.chat(prompt)
        try:
            text = resp.choices[0].message.content
        except Exception:
            text = str(resp)
    except Exception as e:
        text = f"Ошибка при вызове GigaChat: {e}"
    return text, ctx
