import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, users, items, embedding_dim = 64):
        super(Model, self).__init__()

        self.user_embedding = nn.Embedding(users, embedding_dim)
        self.item_embedding = nn.Embedding(items, embedding_dim)

        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_index, item_index):
        user_vector = self.user_embedding(user_index)
        item_vector = self.item_embedding(item_index)

        vector = torch.cat([user_vector, item_vector], dim=1)

        score = self.layers(vector)

        return score.squeeze()