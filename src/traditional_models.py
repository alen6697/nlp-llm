import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


# --------------------------------
# TF-IDF + Logistic Regression
# --------------------------------

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model using TF-IDF features.
    """

    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    model.fit(X_train, y_train)

    return model


def predict_logistic(model, X_test):
    """
    Generate predictions from Logistic Regression model.
    """

    return model.predict(X_test)


# --------------------------------
# Word2Vec + LSTM
# --------------------------------

class LSTMModel(nn.Module):

    def __init__(self, input_dim=100, hidden_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return self.sigmoid(out)


def train_lstm_model(X_train, y_train, epochs=5):

    """
    Train LSTM using Word2Vec sentence embeddings.
    """

    X_train = torch.tensor(X_train).float().unsqueeze(1)

    y_train = torch.tensor(y_train.values).float().unsqueeze(1)

    model = LSTMModel()

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

    return model


def predict_lstm(model, X_test):

    """
    Generate predictions from LSTM model.
    """

    import torch

    X_test = torch.tensor(X_test).float().unsqueeze(1)

    with torch.no_grad():
        preds = model(X_test)

    probs = preds.numpy().flatten()

    preds_binary = (probs > 0.5).astype(int)

    return preds_binary, probs
