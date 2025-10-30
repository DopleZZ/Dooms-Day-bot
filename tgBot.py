import os
import telebot
from typing import Optional
from env_loader import load_dotenv



load_dotenv()


TG_TOKEN = os.getenv("TG_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not TG_TOKEN:
    print("Ошибка: TELEGRAM token (TG_TOKEN) не найден в окружении (.env)")


from GigaChatSDK import rag_answer
from chunks import add_fact


bot = telebot.TeleBot(TG_TOKEN) if TG_TOKEN else None


@bot.message_handler(commands=["start", "help"]) if bot else (lambda f: f)
def start(m):
    bot.reply_to(m, "Я бот по теме Git/тех. знаний — спрашивай. Для добавления факта в группу используйте /learn <текст> или ответьте на сообщение и вызовите /learn")


def _is_group_message(msg) -> bool:
    t = getattr(msg.chat, "type", "private")
    return t in ("group", "supergroup")


@bot.message_handler(commands=["learn"]) if bot else (lambda f: f)
def cmd_learn(m):
    
    fact: Optional[str] = None
    if m.reply_to_message and getattr(m.reply_to_message, "text", None):
        fact = m.reply_to_message.text
    else:
        
        parts = (m.text or "").split(" ", 1)
        if len(parts) > 1:
            fact = parts[1].strip()

    if not fact:
        bot.reply_to(m, "Нечего добавлять. Использование: /learn <текст> или ответ на сообщение + /learn")
        return

    try:
        add_fact(fact)
        bot.reply_to(m, "Факт добавлен в базу знаний.")
    except Exception as e:
        bot.reply_to(m, f"Ошибка при добавлении факта: {e}")


@bot.message_handler(func=lambda m: True) if bot else (lambda f: f)
def handle(m):
    q = (m.text or "").strip()
    if not q:
        return
    try:
        a, ctx = rag_answer(q)
    except Exception as e:
        a = f"Ошибка при формировании ответа: {e}"
    bot.reply_to(m, a)


if __name__ == "__main__" and bot:
    print("Бот запущен. Для остановки прервите процесс.")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)