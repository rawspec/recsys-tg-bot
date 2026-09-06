import os
import pickle
import ssl
import urllib.request
import zipfile
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

def download_and_prepare_data():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    
    if not os.path.exists("ml-100k"):
        print("Скачивание данных MovieLens 100k...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(url, context=ctx) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)

    # Загружаем рейтинги (user_id, item_id, rating)
    ratings = pd.read_csv(
        "ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    # Загружаем названия фильмов
    movies = pd.read_csv(
        "ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1], names=["item_id", "title"]
    )
    return ratings, movies

def main():
    ratings, movies = download_and_prepare_data()

    # Делаем маппинг ID в индексы 0..N
    unique_users = ratings["user_id"].unique()
    unique_items = ratings["item_id"].unique()

    user2idx = {uid: i for i, uid in enumerate(unique_users)}
    item2idx = {iid: i for i, iid in enumerate(unique_items)}
    idx2item = {i: iid for iid, i in item2idx.items()}
    movie_dict = dict(zip(movies["item_id"], movies["title"]))

    rows = ratings["user_id"].map(user2idx)
    cols = ratings["item_id"].map(item2idx)
    values = ratings["rating"].astype(float)

    # Разреженная матрица User-Item
    user_item_matrix = sp.csr_matrix(
        (values, (rows, cols)), shape=(len(unique_users), len(unique_items))
    )

    print("Обучение модели ALS...")
    model = AlternatingLeastSquares(factors=32, iterations=15, regularization=0.1, random_state=42)
    model.fit(user_item_matrix * 10)

    # Сохраняем артефакты
    artifacts = {
        "model": model,
        "user_item_matrix": user_item_matrix,
        "user2idx": user2idx,
        "idx2item": idx2item,
        "movie_dict": movie_dict,
    }
    with open("model_data.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("Готово! Модель обучена и сохранена в model_data.pkl")

if __name__ == "__main__":
    main()