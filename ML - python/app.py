import pandas as pd
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from dataset import Data
from model import Model

parquet_file = "dataset.parquet"
pickle_file = "metadata.pkl"

if not os.path.exists(parquet_file):
    print("Downloading dataset...")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)

    df_clean = df.dropna(subset=["CustomerID"]).copy()
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean["UnitPrice"] > 0)]
    df_clean['StockCode'] = df_clean['StockCode'].astype(str)
    df_clean = df_clean[df_clean['StockCode'].str.match(r'^\d{5}')]
    df_clean = df_clean.sort_values(['CustomerID', 'InvoiceDate'])
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean = df_clean.sort_values(['CustomerID', 'InvoiceDate'])
    
    df_clean.to_parquet(parquet_file, engine='pyarrow')
    
    metadata = {
        "no_of_customers": df_clean['CustomerID'].nunique(),
        "no_of_products": df_clean['StockCode'].nunique(),
        "first_invoice": df_clean['InvoiceDate'].min(),
        "last_invoice": df_clean['InvoiceDate'].max()
    }
    with open(pickle_file, 'wb') as f:
        pickle.dump(metadata, f)
else:
    print("Loading data...")
    df = pd.read_parquet(parquet_file)

    with open(pickle_file, 'rb') as f:
        metadata = pickle.load(f)
    print("Loading finished")

    num_customers = df['CustomerID'].nunique()
    num_products = df['StockCode'].nunique()

    interactions = len(df)

    sparsity = 1 - (interactions / (num_products * num_customers))

    client_interactions = df.groupby('CustomerID').size()
    
    print(f"Average: {client_interactions.mean()}")
    print(f"Max: {client_interactions.max()}")
    print(f"Median: {client_interactions.median()}")
    print(f"Mode: {client_interactions.mode().iloc[0]}")

    plt.figure()
    plt.subplot(1, 2, 1)

    sns.histplot(client_interactions[client_interactions < 500], bins=40, kde=True, legend=False)
    plt.title("Distribution fo products per user")
    plt.xlabel("No of products")
    plt.ylabel("No of clients")

    plt.subplot(1, 2, 2)
    top_products = df['StockCode'].value_counts().head(10)
    sns.barplot(hue=top_products.index, y=top_products.values, palette='viridis')
    plt.title("Top 10 Products")
    plt.xlabel("Product Code")
    plt.ylabel("No of products")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

    frequent_clients = client_interactions[client_interactions >= 3].index
    df_dl = df[df['CustomerID'].isin(frequent_clients)].copy()

    df_dl = df_dl.sort_values(['CustomerID', 'InvoiceDate'])

    df_dl['Ordering'] = df_dl.groupby('CustomerID').cumcount(ascending=False)

    test_set = df_dl[df_dl['Ordering'] == 0].copy()
    val_set = df_dl[df_dl['Ordering'] == 1].copy()
    train_set = df_dl[df_dl['Ordering'] > 1].copy()

    test_set = test_set.drop(columns=['Ordering'])
    val_set = val_set.drop(columns=['Ordering'])
    train_set = train_set.drop(columns=['Ordering'])

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df_dl['user_idx'] = user_encoder.fit_transform(df_dl['CustomerID'])
    df_dl['item_idx'] = item_encoder.fit_transform(df_dl['StockCode'])

    train_set['user_idx'] = user_encoder.transform(train_set['CustomerID'])
    train_set['item_idx'] = item_encoder.transform(train_set['StockCode'])

    val_set['user_idx'] = user_encoder.transform(val_set['CustomerID'])
    val_set['item_idx'] = item_encoder.transform(val_set['StockCode'])

    test_set['user_idx'] = user_encoder.transform(test_set['CustomerID'])
    test_set['item_idx'] = item_encoder.transform(test_set['StockCode'])

    num_users = df_dl['user_idx'].nunique()
    num_items = df_dl['item_idx'].nunique()

    print(f"Unique users: {num_users}")
    print(f"Unique products: {num_items}")

    train_dataset = Data(
        users=train_set['user_idx'].values,
        items=train_set['item_idx'].values,
        total_items=num_items,
        negative_samples=4
    )

    data_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True)

    model = Model(users=num_users, items=num_items, embedding_dim=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.015)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in data_loader:
            users = batch['user'].to(device)
            item_pos = batch['positive_item'].to(device)
            item_neg = batch['negative_items'].to(device)
            batch_size = users.size(0)
            num_negatives = item_neg.size(1)

            pos_scores = model(users, item_pos)
            pos_labels = torch.ones_like(pos_scores)

            users_expanded = users.unsqueeze(1).repeat(1, num_negatives).view(-1)
            item_neg_flat = item_neg.view(-1)

            neg_scores = model(users_expanded, item_neg_flat)
            neg_labels = torch.zeros_like(neg_scores)

            all_scores = torch.cat([pos_scores, neg_scores])
            all_labels = torch.cat([pos_labels, neg_labels])

            loss = criterion(all_scores, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    print("\n Training complete!")

    model.eval()

    K = 10
    hits = 0
    total_users = 0

    print("🔍 Starting Final Evaluation (Hit Ratio @ 10)...")

    test_users = test_set['user_idx'].unique()[:500]

    with torch.no_grad():
        for u in test_users:
            item_real = test_set[test_set['user_idx'] == u]['item_idx'].values[0]

            items_fake = random.sample(range(num_items), 99)

            if item_real in items_fake:
                items_fake.remove(item_real)
                items_fake.append(random.randint(0, num_items - 1))

            items_candidates = [item_real] + items_fake

            user_tensor = torch.tensor([u] * 100).to(device)
            items_tensor = torch.tensor(items_candidates).to(device)

            scores = model(user_tensor, items_tensor).cpu().numpy()

            top_k_indices = scores.argsort()[-K:]

            if 0 in top_k_indices:
                hits += 1

            total_users += 1

    hit_ratio = (hits / total_users) * 100
    print(f"\n🏆 Final Result - HR@{K}: {hit_ratio:.2f}%")
    print(f"👉 For {hit_ratio:.0f}% of users, the model ranked the correct item in the Top {K} recommendations!")

    model_file_name = "recommendation_model.pth"

    torch.save(model.state_dict(), model_file_name)
    
    user_history = df_dl.groupby('CustomerID')['StockCode'].apply(set).to_dict()

    user_to_index = {user_id: index for index, user_id in enumerate(user_encoder.classes_)}
    index_to_item = {index: item_id for index, item_id in enumerate(item_encoder.classes_)}

    api_metadata = {
        "user_to_index" : user_to_index,
        "index_to_item" : index_to_item,
        "user_history" : user_history,
        "num_users" : len(user_to_index),
        "num_items" : len(index_to_item)
    }

    with open("api_metadata.pkl", "wb") as f:
        pickle.dump(api_metadata, f)
    






    

    






