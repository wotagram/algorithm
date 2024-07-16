import numpy as np


def analyze_clusters(posts_dataframe, ratings_dataframe, train_loader, kmeans):
    posts_contents = posts_dataframe.set_index('id')['content'].to_dict()

    for cluster in range(20):
        print("Cluster #{}".format(cluster))
        posts = []
        for postidx in np.where(kmeans.labels_ == cluster)[0]:
            # Ensure postid is a string
            postid = str(train_loader.idx2postid[postidx])
            if postid in posts_contents:
                rat_count = ratings_dataframe.loc[ratings_dataframe['post_id'] == postid]['post_id'].count(
                )
                posts.append((posts_contents[postid], rat_count))
            else:
                print(f"Post ID {postid} not found in posts_contents.")

        # Sort and print top posts in the cluster by rat_count
        for post in sorted(posts, key=lambda tup: tup[1], reverse=True)[:20]:
            print("\t", post[0])
            print("\t", "="*80)
