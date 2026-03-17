from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
import numpy as np
import pickle
from model import Model

app = FastAPI()

df_clean = pd.read_parquet("dataset.parquet")

with open("api_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

user_to_index = metadata["user_to_index"]

index_to_item = metadata["index_to_item"]
user_history = metadata["user_history"]
num_users = metadata["num_users"]
num_items = metadata["num_items"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(users=num_users, items=num_items, embedding_dim=64)

model.load_state_dict(torch.load("recommendation_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

@app.get("/catalog")
def get_catalog():
    
    catalog = df_clean[['StockCode', 'Description', 'UnitPrice']].drop_duplicates('StockCode')
    
    
    catalog = catalog.rename(columns={
        'StockCode': 'code',
        'Description': 'name',
        'UnitPrice': 'price'
    })
    
    
    return catalog.head(1000).to_dict(orient='records')


@app.get("/recommandations/{customer_id}")
def get_recommandations(customer_id: int):
    if customer_id not in user_to_index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = user_to_index[customer_id]
    bought_items = user_history.get(customer_id, set())

    all_products_idx = torch.arange(num_items).to(device)
    user_tensor = torch.tensor([user_idx] * num_items).to(device)

    with torch.no_grad():
        scores = model(user_tensor, all_products_idx).cpu().numpy()

    ordered_items_idx = scores.argsort()[::-1]

    recommandations = []
    for idx in ordered_items_idx:
        stock_code = index_to_item[idx]

        if stock_code not in bought_items:
            recommandations.append(stock_code)

        if len(recommandations) == 10:
            break
    
    return{
        "customerID" : customer_id,
        "recommandations" : recommandations
    }

