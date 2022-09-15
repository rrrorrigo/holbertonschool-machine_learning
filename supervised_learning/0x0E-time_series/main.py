#!/usr/bin/env python3
"""Main file to execute scripts and keras model"""


import pandas as pd
clean_data = __import__('preprocess_data').clean_data
save_data = __import__('preprocess_data').save_data
preprocess_data = __import__('preprocess_data').preprocess_data
Forecasting = __import__('forecast_btc').Forecasting


if __name__ == '__main__':
    path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    df = pd.read_csv(path)
    data = clean_data(df)
    X_train, X_validate, X_test, Y_train, Y_val, Y_test = preprocess_data(data)
    batch_size = 256

    forecasting = Forecasting(X_train, X_validate, X_test, Y_train, Y_val, Y_test, batch_size)

    model = forecasting.create()

    model, hist = forecasting.train(model=model)

    forecasting.plot_0(df['Close'], 'BTC: Price at Close vs. Timestamp')

    # Plot the model loss results
    forecasting.plot_1(hist, 'Training / Validation Losses from History')

    string = 'Predictions over a {} x 24h Timeframe (Batch {})'
    # Make a single-step price prediction following 24h of data
    window_num = 0
    for batch_num, (x, y) in enumerate(forecasting.val_dataset.take(3)):
          title = string.format(window_num, batch_num)
          forecasting.plot_2(x[window_num, :, -2].numpy(),
                 y[window_num].numpy(),
                 model.predict(x)[window_num],
                 title)

    # Make predictions over "batch_size" x 24h timeframes
    for batch_num, (x, y) in enumerate(forecasting.val_dataset.take(3)):
        title = string.format(batch_size, batch_num)
        forecasting.plot_3(y.numpy(),
               model.predict(x).reshape(-1),
               title)
        batch_num += 1
