import torch
from torch.utils.data import Dataset


class DataHandler(Dataset):
    def __init__(self, ratings_dataframe):
        self.ratings = ratings_dataframe.copy()

        users = ratings_dataframe.user_id.unique()
        posts = ratings_dataframe.post_id.unique()

        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.postid2idx = {o: i for i, o in enumerate(posts)}

        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2postid = {i: o for o, i in self.postid2idx.items()}

        self.ratings.post_id = ratings_dataframe.post_id.apply(
            lambda x: self.postid2idx[x])
        self.ratings.user_id = ratings_dataframe.user_id.apply(
            lambda x: self.userid2idx[x])

        self.x = self.ratings[['user_id', 'post_id']].values
        self.y = self.ratings['rating'].values

        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.ratings)
