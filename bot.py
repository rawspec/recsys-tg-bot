import asyncio
import pickle
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command

# Вставь свой токен от BotFather сюда или задай через переменную окружения BOT_TOKEN
TOKEN = os.getenv("BOT_TOKEN", "YOUR_TOKEN__")

# Загружаем сохраненную модель
print("Загрузка модели...")
with open("model_data.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
user_item_matrix = data["user_item_matrix"]
user2idx = data["user2idx"]
idx2item = data["idx2item"]
movie_dict = data["movie_dict"]

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я ML-бот рекомендательной системы.\n\n"
        "Отправь мне ID существующего пользователя (число от 1 до 943), "
        "и я выдам топ-5 персональных рекомендаций фильмов на основе ALS-коллаборативной фильтрации.\n\n"
        "Пример: напиши просто `10` или `42`"
    )

@dp.message()
async def recommend_handler(message: types.Message):
    text = message.text.strip()
    if not text.isdigit():
        await message.answer("Пожалуйста, отправь число — ID пользователя (от 1 до 943).")
        return

    uid = int(text)
    if uid not in user2idx:
        await message.answer("Пользователь с таким ID не найден в обучающей выборке (введи от 1 до 943).")
        return

    u_idx = user2idx[uid]
    # Вызов ALS-рекомендаций: отбираем топ-5
    item_indices, scores = model.recommend(
        u_idx, user_item_matrix[u_idx], N=5, filter_already_liked_items=True
    )

    response = [f"🎬 **Топ-5 рекомендаций для пользователя #{uid}:**\n"]
    for rank, item_idx in enumerate(item_indices, 1):
        raw_id = idx2item[item_idx]
        title = movie_dict.get(raw_id, "Неизвестный фильм")
        response.append(f"{rank}. {title}")

    await message.answer("\n".join(response), parse_mode="Markdown")

async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())