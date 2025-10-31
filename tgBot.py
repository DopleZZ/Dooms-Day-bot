import os
import telebot
from typing import Optional
from env_loader import load_dotenv
import threading



load_dotenv()


TG_TOKEN = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not TG_TOKEN:
    print("Ошибка: TELEGRAM token (TG_TOKEN) не найден в окружении (.env)")


from GigaChatSDK import rag_answer
from chunks import add_fact


bot = telebot.TeleBot(TG_TOKEN) if TG_TOKEN else None

CHAT_COUNTERS = {}
CHAT_THRESHOLDS = {}
DEFAULT_THRESHOLD = 20
AUTO_LEARN = True
MIN_LEARN_LEN = 30


@bot.message_handler(commands=["start", "help"]) if bot else (lambda f: f)
def start(m):
    bot.reply_to(m, "и че надо")


def _is_group_message(msg) -> bool:
    t = getattr(msg.chat, "type", "private")
    return t in ("group", "supergroup")


@bot.message_handler(func=lambda m: True) if bot else (lambda f: f)
def handle(m):
    text = (m.text or "").strip()
    if not text:
        return

    cid = getattr(m.chat, "id", None)
    is_group = _is_group_message(m)

    if text.startswith("/"):
        return

    if is_group:
        if cid is None:
            return
        CHAT_COUNTERS[cid] = CHAT_COUNTERS.get(cid, 0) + 1
        threshold = CHAT_THRESHOLDS.get(cid, DEFAULT_THRESHOLD)
        if CHAT_COUNTERS[cid] >= threshold:
            try:
                a, ctx = rag_answer(text)
            except Exception as e:
                a = f"Ошибка при формировании ответа: {e}"
            bot.send_message(cid, a)
            CHAT_COUNTERS[cid] = 0
        if AUTO_LEARN:
            def _worker(t):
                try:
                    s = t.strip()
                    wants = False
                    if len(s) >= MIN_LEARN_LEN:
                        wants = True
                    elif s.startswith("[") and "]" in s:
                        wants = True
                    elif ":" in s and len(s.split(":", 1)[0].strip()) <= 60:
                        wants = True
                    if wants:
                        add_fact(s)
                except Exception:
                    pass
            threading.Thread(target=_worker, args=(text,), daemon=True).start()
        return

    try:
        a, ctx = rag_answer(text)
    except Exception as e:
        a = f"Ошибка при формировании ответа: {e}"
    bot.reply_to(m, a)
    if AUTO_LEARN:
        def _worker2(t):
            try:
                s = t.strip()
                wants = False
                if len(s) >= MIN_LEARN_LEN:
                    wants = True
                elif s.startswith("[") and "]" in s:
                    wants = True
                elif ":" in s and len(s.split(":", 1)[0].strip()) <= 60:
                    wants = True
                if wants:
                    add_fact(s)
            except Exception:
                pass
        threading.Thread(target=_worker2, args=(text,), daemon=True).start()


@bot.message_handler(commands=["set_n"]) if bot else (lambda f: f)
def cmd_set_n(m):
    parts = (m.text or "").split()
    if len(parts) < 2:
        bot.reply_to(m, "Использование: /set_n <N>")
        return
    try:
        n = int(parts[1])
        if n <= 0:
            raise ValueError()
    except Exception:
        bot.reply_to(m, "N должен быть положительным целым числом")
        return
    cid = getattr(m.chat, "id", None)
    if cid is None:
        bot.reply_to(m, "не удалось определить чат")
        return
    CHAT_THRESHOLDS[cid] = n
    CHAT_COUNTERS[cid] = 0
    bot.reply_to(m, f"порог сообщений: {n}")


@bot.message_handler(commands=["get_n"]) if bot else (lambda f: f)
def cmd_get_n(m):
    cid = getattr(m.chat, "id", None)
    if cid is None:
        bot.reply_to(m, "не удалось определить чат")
        return
    n = CHAT_THRESHOLDS.get(cid, DEFAULT_THRESHOLD)
    bot.reply_to(m, f"текущий порог для ответов: {n} сообщений")


@bot.message_handler(commands=["reset_counter"]) if bot else (lambda f: f)
def cmd_reset_counter(m):
    cid = getattr(m.chat, "id", None)
    if cid is None:
        bot.reply_to(m, "не удалось определить чат")
        return
    CHAT_COUNTERS[cid] = 0
    bot.reply_to(m, "счётчик пропущенных сообщений сброшен")


if __name__ == "__main__" and bot:
    print("Бот запущен. Для остановки прервите процесс.")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)