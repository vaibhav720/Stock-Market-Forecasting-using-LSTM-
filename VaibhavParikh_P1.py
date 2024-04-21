
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class Stock_Price_Forecasting_LSTM:
    def __init__(self, stock_data_path):
        self.stock_data = pd.read_csv(stock_data_path)
        print(self.stock_data.head())
        self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_training_data = None
        self.X_testing_data = None
        self.y_training_data = None
        self.y_test_data = None
        self.lstm_model = None

    def stock_data_cleaning(self, time_steps):
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
        self.stock_data.sort_values('Date', inplace=True)
        daily_closing_price = self.stock_data['Close'].values.reshape(-1, 1)
        scaled_daily_closing_price = self.min_max_scaler.fit_transform(daily_closing_price)
        X_data, y_data = self.load_stock_dataset(scaled_daily_closing_price, time_steps)
        X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
        train_size = int(len(X_data) * 0.97)
        self.X_training_data, self.X_testing_data = X_data[:train_size], X_data[train_size:]
        self.y_training_data, self.y_test_data = y_data[:train_size], y_data[train_size:]
        print("After cleaning X training data"+str(self.X_training_data))

    def load_stock_dataset(self, data, time_steps):
        X_data, y_data = [], []
        for i in range(len(data) - time_steps):
            X_data.append(data[i:(i + time_steps), 0])
            y_data.append(data[i + time_steps, 0])
        return np.array(X_data), np.array(y_data)

    def lstm_stock_model(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=132, return_sequences=True, input_shape=(self.X_training_data.shape[1], 1)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(LSTM(units=110, return_sequences=False))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_model = lstm_model
        print(self.lstm_model.summary())

    def lstm_model_training(self, epochs=80, batch_size=16, validation_split=0.1):
        lstm_stock_market_trained_model = self.lstm_model.fit(self.X_training_data, self.y_training_data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return lstm_stock_market_trained_model

    def rsme_mae(self):
        lstm_train_prediction = self.lstm_model.predict(self.X_training_data)
        lstm_test_predictions = self.lstm_model.predict(self.X_testing_data)
        lstm_train_prediction = self.min_max_scaler.inverse_transform(lstm_train_prediction)
        y_training_data = self.min_max_scaler.inverse_transform([self.y_training_data])
        lstm_test_predictions = self.min_max_scaler.inverse_transform(lstm_test_predictions)
        y_test_data = self.min_max_scaler.inverse_transform([self.y_test_data])

        trianed_data_rsme_result = np.sqrt(mean_squared_error(y_training_data[0], lstm_train_prediction[:, 0]))
        test_data_rsme_result = np.sqrt(mean_squared_error(y_test_data[0], lstm_test_predictions[:, 0]))
        trianed_data_mae_result = mean_absolute_error(y_training_data[0], lstm_train_prediction[:, 0])
        tested_data_mae_result = mean_absolute_error(y_test_data[0], lstm_test_predictions[:, 0])

        return trianed_data_rsme_result, test_data_rsme_result, trianed_data_mae_result, tested_data_mae_result, y_training_data[0], lstm_train_prediction[:, 0], y_test_data[0], lstm_test_predictions[:, 0]

    def graph_training_loss_validation_loss(self, lstm_predicted_stock):
        plt.plot(lstm_predicted_stock.history['loss'], label='Training Loss')
        plt.plot(lstm_predicted_stock.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss and Validation Loss after training')
        plt.legend()
        plt.show()

    def lstm_stock_market_forecasting_plot(self, test_actual, test_predicted):
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Date'][-len(test_actual):], test_actual, label='Actual Test plot')
        plt.plot(self.stock_data['Date'][-len(test_predicted):], test_predicted, label='Predicted Test plot')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Prediction and Actual plot')
        plt.legend()
        plt.show()

    def five_year_stock_price_plot(self, test_actual, test_predicted):
        plt.figure(figsize=(12, 6))
        plt.plot(self.stock_data['Date'][-(5 * 365):], self.stock_data['Close'][-(5 * 365):], label='Stock Prices of Last 5 years')
        plt.plot(self.stock_data['Date'][-len(test_actual):], test_actual, label='Actual Test Plot')
        plt.plot(self.stock_data['Date'][-len(test_predicted):], test_predicted, label='Predicted Test Plot')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Stock Market Forecasting')
        plt.legend()
        plt.show()

stock_forecaster = Stock_Price_Forecasting_LSTM('UBER.csv')

stock_forecaster.stock_data_cleaning(time_steps=60)

stock_forecaster.lstm_stock_model()

lstm_predicted_stock = stock_forecaster.lstm_model_training(epochs=110, batch_size=16, validation_split=0.1)

stock_forecaster.graph_training_loss_validation_loss(lstm_predicted_stock)
trained_data_rmse_result, tested_data_rmse_result, trained_data_mae_result, tested_data_mae_result, train_actual, train_predicted, test_actual, test_predicted = stock_forecaster.rsme_mae()
print(f'Training data root mean square error (RMSE): {trained_data_rmse_result}')
print(f'Testing data root mean square error (RMSE): {tested_data_rmse_result}')
print(f'Training data Mean Absolute Error (MAE): {trained_data_mae_result}')
print(f'Testing data Mean Absolute Error (MAE): {tested_data_mae_result}')

stock_forecaster.lstm_stock_market_forecasting_plot(test_actual, test_predicted)

stock_forecaster.five_year_stock_price_plot(test_actual, test_predicted)