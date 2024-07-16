import pandas as pd


def load_data(posts_path, ratings_path):
    posts_dataframe = pd.read_csv(posts_path)
    ratings_dataframe = pd.read_csv(ratings_path)

    print("The dimensions of posts dataframe are ", posts_dataframe.shape, "\n")

    print(posts_dataframe.head())

    print("The dimensions of ratings dataframe are ",
          ratings_dataframe.shape, "\n")

    print(ratings_dataframe.head())

    number_of_users = len(ratings_dataframe['user_id'].unique())
    number_of_posts = len(ratings_dataframe['post_id'].unique())

    print('Number of unique users = ' + str(number_of_users) + ' \n')
    print('Number of unique posts = ' + str(number_of_posts) + ' \n')
    print("The full rating matrix will have:",
          number_of_users*number_of_posts, 'elements.')
    print("Therefore: ", len(ratings_dataframe) / (number_of_users *
          number_of_posts) * 100, '% of the matrix is filled.')

    return posts_dataframe, ratings_dataframe
