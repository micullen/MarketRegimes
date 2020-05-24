from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import cm
from pandas import DataFrame
from typing import List, Tuple
import numpy as np

class CreateModel:
    """Class to create clustering models for volatility and trend."""

    def __init__(self, n_components: int, n_clusters: int, metrics: List, dataframe: DataFrame) -> None:
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.dataframe = dataframe
        self.metrics = metrics
        self.trim_df = dataframe[metrics]

    def prep_data(self, df: DataFrame) -> DataFrame:
        """Normalise both training and testing data"""
        X = df.values
        pca = PCA(self.n_components, whiten=True)
        X_transformed = pca.fit_transform(X)
        return X_transformed

    def split_train_test(self, data: DataFrame, test_ratio: float) -> Tuple:
        """Randomly split data into test and training data"""
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    def handle_model(self):
        """This function calls all the required functions to make the model and returns the model"""
        train_df, test_df = self.split_train_test(self.trim_df, 0.5)
        train_X = self.prep_data(train_df)
        test_X = self.prep_data(test_df)

        return self.cluster(train_X)

    def cluster(self, X: DataFrame):
        """Create model from transformed data"""
        model = KMeans(n_clusters=self.n_clusters).fit(X)
        return model

    def plot_kmeans_clusters(self, model, df: DataFrame, X: DataFrame) -> None:
        """Plot closing price in each regime"""
        # Predict the hidden states array
        hidden_states = model.predict(X)

        # Colours for regimes
        colours = cm.rainbow(
            np.linspace(0, 1, model.n_clusters)
        )
        i = 0
        # for i, (ax, colour) in enumerate(zip(axs, colours)):
        for colour in colours:
            mask = hidden_states == i
            plt.plot(df.index[mask], df["close"][mask], ".", linestyle='none', c=colour)
            i += 1

        plt.show()