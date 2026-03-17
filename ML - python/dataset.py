import torch
import random
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, users, items, total_items, negative_samples = 0):
        self.users = users
        self.items = items
        self.total_items = total_items
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        positive_item = self.items[index]

        negative_items = []

        for _ in range(self.negative_samples):
            item = random.randint(0, self.total_items - 1)
            negative_items.append(item)

        return {
            'user': torch.tensor(user, dtype=torch.long),
            'positive_item': torch.tensor(positive_item, dtype=torch.long),
            'negative_items': torch.tensor(negative_items, dtype=torch.long )
        }


