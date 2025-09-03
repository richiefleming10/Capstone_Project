import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pandas as pd 

class EntropyConfidenceModel:

    def __init__(self, lookback, windows):
        self.lookback = lookback
        self.windows = windows
        self.model = None

    def compute_entropy(self, data):
        histogram_data = np.histogram(data, bins=int(np.ceil(np.log2(len(data) + 1))))[0]
        prob_dist = histogram_data / histogram_data.sum()
        entropy = -np.sum(prob_dist[prob_dist > 0] * np.log2(prob_dist[prob_dist > 0]))
        return entropy

    def build_lstm_attn(self, input_shape):
        seq_in = layers.Input(shape=input_shape)
        lstm_out = layers.LSTM(32, return_sequences=False)(seq_in)
        last_step = layers.Lambda(lambda t: t[:, -1, :])(seq_in)
        att_dense = layers.Dense(input_shape[-1], name="att_dense")(last_step)
        att_soft = layers.Activation("softmax", name="attn")(att_dense)
        context = layers.Lambda(
            lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True)
        )([last_step, att_soft])
        merged = layers.Concatenate()([lstm_out, context])
        out = layers.Dense(1, activation="sigmoid")(merged)

        model = models.Model(seq_in, out)
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")]
        )
        self.model = model

    def fit(self, X, y, epochs=100, batch_size=32):
        if self.model is None:
            self.build_lstm_attn((self.lookback, len(self.windows)))
        es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=0
        )

    def predict_confidence(self, recent_data_windows):
        if self.model is None:
            raise ValueError("Model must be trained before predictions.")
        confidence = self.model.predict(recent_data_windows, verbose=0).flatten()
        return confidence

    def get_attention_distribution(self, X):
        att_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer("attn").output
        )
        att_values = att_model.predict(X)
        return att_values

    def dynamic_entropy(self, data_windows):
        attention_values = self.get_attention_distribution(data_windows)
        optimal_window_index = np.argmax(attention_values.mean(axis=0))
        optimal_window_size = self.windows[optimal_window_index]
        recent_data = data_windows[:, -optimal_window_size:, optimal_window_index]
        entropy_values = np.array([self.compute_entropy(window) for window in recent_data])
        return entropy_values
    
    def generate_confidence_scores(self, price_series):
        returns = price_series.pct_change().dropna().values
        n_samples = len(returns) - self.lookback
        data_windows = np.zeros((n_samples, self.lookback, len(self.windows)))

        for idx, window in enumerate(self.windows):
            for i in range(n_samples):
                start = i
                end = i + self.lookback
                if end - window >= 0:
                    window_data = returns[end - window:end]
                    entropy = self.compute_entropy(window_data)
                    data_windows[i, -window:, idx] = entropy

        confidence_scores = self.predict_confidence(data_windows)
        confidence_series = np.concatenate([np.full(self.lookback + 1, np.nan), confidence_scores])
        return confidence_series



if __name__ == '__main__':
    # Load dataset (Assume CSV file named 'nvidia_prices.csv' with columns 'Date' and 'Price')
    df = pd.read_csv('nvidia_prices.csv', parse_dates=['Date'])
    prices = df['Price']

    # Define parameters
    lookback = 30
    windows = [20, 60, 90, 130, 252]

    # Initialize the model
    ecm = EntropyConfidenceModel(lookback, windows)

    # Prepare data for training (here simplified for demonstration)
    returns = prices.pct_change().dropna()
    X_train = np.random.rand(len(returns) - lookback, lookback, len(windows))  # Replace with actual data
    y_train = np.random.randint(0, 2, len(returns) - lookback)  # Replace with actual labels

    # Fit the model
    ecm.fit(X_train, y_train, epochs=5)

    # Generate confidence scores
    confidence_scores = ecm.generate_confidence_scores(prices)

    # Combine with original data and display
    df['Confidence'] = confidence_scores
    print(df.head(40))