if __name__ == "__main__":
    try:
        from GigaChatSDK import rag_answer
        ans, ctx = rag_answer("у кого биг бадонки и от кого может вонять?")
        print("Ответ:\n", ans)
        print("\nТоп-контекст:")
        for t, s in ctx:
            print(f"- score={s:.3f}: {t[:120]}...")
    except Exception as e:
        print("Ошибка при запуске теста:", e)