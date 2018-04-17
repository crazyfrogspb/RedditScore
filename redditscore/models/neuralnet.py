import keras
from keras.initializers import RandomNormal
from keras.layers import Activation, Dense, Embedding
from keras.models import Sequential

from . import redditmodel


class MLPModel(redditmodel.RedditModel):
    def __init__(self, n_layers=1, layer_sizes=32, dim=100, embeddings=None, random_state=24):
        super().__init__(random_state=random_state)
        if not isinstance(n_layers, int):
            raise ValueError("n_layers must be an integer")
        if not isinstance(dim, int):
            raise ValueError("dim must be an integer")

        if isinstance(layer_sizes, int):
            self.layer_sizes = [layer_sizes] * n_layers
        elif isinstance(layer_sizes, list):
            if len(layer_sizes) != n_layers:
                raise ValueError(
                    'Length of layer_sizes must be equal to n_layers')
            else:
                self.layer_sizes = layer_sizes
        else:
            raise ValueError("layer_sizes must be an integer or a list")

        self.n_layers = n_layers
        self.dim = dim
        self.embeddings = embeddings
        self.activation = activation
        self.dropout = dropout

        self._vocab_size = 0

    def _build_vocab(self):
        pass

    def _build_model(self):
        model = Sequential()

        if self.embeddings is None:
            model.add(Embedding(self._vocab_size, self.dim,
                                embeddings_initializer=RandomNormal(seed=self.random_state)))
        else:
            model.add(Embedding(self._vocab_size, self.dim,
                                weights=[self.weights], trainable=False))

        weights = [weights], trainable = train_embeddings
        model.add(Dense(layer_sizes[0], input_shape=(dim, )))
        for i in range(n_layers):

            model.add(Dense(layer_sizes[i]))


class CNNModel(redditmodel.RedditModel):
    pass


class LSTMModel(redditmodel.RedditModel):
    pass
