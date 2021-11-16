import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
from tqdm import tqdm

difficulty_flow = {}
all_r2 = 0  # calculating the average r2 score

original = pd.read_csv("../../data/interactions.csv")

# extract the winning data
interactions = original.loc[original.win == 1].reset_index(drop=True)

# only preserve the first winning record
retry_data = interactions.groupby(["user_id", "level_id"]).head(1)

# calculate average retry time for each level
new_global_retry = retry_data.groupby(["level_id"]).retry_time.mean().reset_index()
new_global_retry.rename(columns={"retry_time": "global_retry_time"}, inplace=True)

# Neighbor = 5

user_ids = list(retry_data.user_id.unique())
for user_id in tqdm(user_ids):
    id_data = retry_data.loc[retry_data.user_id == user_id]
    # if two levels get the same global retry time
    id_data = id_data.merge(new_global_retry, on=["level_id"], how="left").groupby(["global_retry_time"]).agg({"retry_time": "mean"}).reset_index()
    x = id_data.global_retry_time.to_numpy()
    y = id_data.retry_time.to_numpy()
    model = LinearRegression(copy_X=True)
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    pred_y = model.predict(x.reshape(-1, 1))
    r2 = r2_score(y.reshape(-1, 1), pred_y)
    all_r2 += r2
    difficulty_flow[str(user_id)] = [float(model.coef_[0]), float(model.intercept_[0])]

print(r2 / len(user_ids))

with open("../../data/Difficulty_Flow/myoutput.json", "w") as F:
    json.dump(difficulty_flow, F)
