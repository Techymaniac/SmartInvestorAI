import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def train_lstm(df):

    if len(df) < 60:
        return None, None   # 🔥 FIX

    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    if len(X) == 0:
        return None, None   # 🔥 EXTRA SAFETY

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler


def predict_next(model, scaler, df):

    if model is None or scaler is None:
        return None   # 🔥 IMPORTANT

    data = df[['Close']].values
    scaled_data = scaler.transform(data)

    last_60 = scaled_data[-60:]
    last_60 = last_60.reshape(1,60,1)

    pred = model.predict(last_60)
    pred_price = scaler.inverse_transform(pred)

    return pred_price[0][0]