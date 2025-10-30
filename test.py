if __name__ == "__main__":
    try:
        from GigaChatSDK import rag_answer
        prmt = "у кого биг бадонки и от кого может вонять?"
        ans, ctx = rag_answer(prmt)
        print("входные данные: ", prmt, "\n", "Ответ:\n", ans)
        print("\nТоп-контекст:")
        for t, s in ctx:
            print(f"- score={s:.3f}: {t[:120]}...")
    except Exception as e:
        print("Ошибка при запуске теста:", e)