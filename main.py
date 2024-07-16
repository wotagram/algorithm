from sklearn.cluster import KMeans
import torch
from src.cluster_analysis import analyze_clusters
from src.data_handler import DataHandler
from src.data_loader import load_data
from src.model import MatrixFactorization
from src.train import train_model
from torch.utils.data import DataLoader

if __name__ == "__main__":
    posts_dataframe, ratings_dataframe = load_data(
        'data/posts.csv', 'data/ratings.csv')

    number_of_users = len(ratings_dataframe['user_id'].unique())
    number_of_posts = len(ratings_dataframe['post_id'].unique())

    model = MatrixFactorization(
        number_of_users, number_of_posts, n_factors=100)

    print(model)

    train_set = DataHandler(ratings_dataframe)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    print(train_loader)

    model = train_model(model, train_loader, num_epochs=128)
    torch.save(model, 'model.pth')

    trained_post_embeddings = model.item_factors.weight.data.cpu().numpy()
    kmeans = KMeans(n_clusters=20, random_state=0).fit(trained_post_embeddings)

    analyze_clusters(posts_dataframe, ratings_dataframe, train_set, kmeans)
