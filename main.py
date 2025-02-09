import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
csv_file_path = 'data_aqi.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file_path)

# Convert 'Start' to datetime and set it as the index
data['Start'] = pd.to_datetime(data['Start'], format='%m/%d/%Y %H:%M')
data.set_index('Start', inplace=True)

# Select the columns for the pollutants
pollutants = ['Value SO2', 'Value PM10', 'Value NO2', 'Value PM2.5']

# Ensure the data is sorted by date
data.sort_index(inplace=True)

# Use data at hourly frequency
hourly_data = data[pollutants]

# Check for missing values and handle them
hourly_data = hourly_data.apply(pd.to_numeric, errors='coerce')
hourly_data.fillna(method='ffill', inplace=True)

# Verify no missing values remain
print("\nNumber of missing values in each pollutant column after handling infs:")
print(hourly_data.isna().sum())

# Differencing to remove trend and seasonality for hourly data
diff_data_hourly = hourly_data.diff().dropna()

# Fit the VAR model on the differenced hourly dataset
model_hourly = VAR(diff_data_hourly)
model_fit_hourly = model_hourly.fit(maxlags=15, ic='aic')

# Make hourly predictions for the next 48 hours (2 days)
forecast_steps_hourly = 48
lag_order_hourly = model_fit_hourly.k_ar
forecast_input_hourly = diff_data_hourly.values[-lag_order_hourly:]
hourly_predictions = model_fit_hourly.forecast(y=forecast_input_hourly, steps=forecast_steps_hourly)

# Convert differenced predictions back to original scale for hourly data
last_values_hourly = hourly_data.iloc[-1]
hourly_predictions_cumsum = hourly_predictions.cumsum(axis=0)
hourly_predictions = hourly_predictions_cumsum + last_values_hourly.values

# Generate future timestamps for the hourly predictions
last_date_hourly = hourly_data.index[-1]
future_dates_hourly = [last_date_hourly + pd.Timedelta(hours=i) for i in range(1, forecast_steps_hourly + 1)]
hourly_predictions_df = pd.DataFrame(hourly_predictions, index=future_dates_hourly, columns=pollutants)

# Make SO2 predictions absolute
hourly_predictions_df['Value SO2'] = hourly_predictions_df['Value SO2'].abs()

# Resample the data to daily frequency for daily predictions
daily_data = hourly_data.resample('D').mean()

# Differencing to remove trend and seasonality for daily data
diff_data_daily = daily_data.diff().dropna()

# Fit the VAR model on the differenced daily dataset
model_daily = VAR(diff_data_daily)
model_fit_daily = model_daily.fit(maxlags=15, ic='aic')

# Make daily predictions for the next 10 days
forecast_steps_daily = 10
lag_order_daily = model_fit_daily.k_ar
forecast_input_daily = diff_data_daily.values[-lag_order_daily:]
daily_predictions = model_fit_daily.forecast(y=forecast_input_daily, steps=forecast_steps_daily)

# Convert differenced predictions back to original scale for daily data
last_values_daily = daily_data.iloc[-1]
daily_predictions_cumsum = daily_predictions.cumsum(axis=0)
daily_predictions = daily_predictions_cumsum + last_values_daily.values

# Generate future timestamps for the daily predictions
last_date_daily = daily_data.index[-1]
future_dates_daily = [last_date_daily + pd.Timedelta(days=i) for i in range(1, forecast_steps_daily + 1)]
daily_predictions_df = pd.DataFrame(daily_predictions, index=future_dates_daily, columns=pollutants)

# Make SO2 predictions absolute
daily_predictions_df['Value SO2'] = daily_predictions_df['Value SO2'].abs()


# Function to calculate AQI for a given pollutant
def calculate_aqi(concentration, breakpoints):
    for bp in breakpoints:
        if bp['low'] <= concentration <= bp['high']:
            aqi = ((bp['I_high'] - bp['I_low']) / (bp['high'] - bp['low'])) * (concentration - bp['low']) + bp['I_low']
            return round(aqi)
    return None


# Define breakpoints for AQI calculation
breakpoints_pm25 = [
    {'low': 0.0, 'high': 12.0, 'I_low': 0, 'I_high': 50},
    {'low': 12.1, 'high': 35.4, 'I_low': 51, 'I_high': 100},
    {'low': 35.5, 'high': 55.4, 'I_low': 101, 'I_high': 150},
    {'low': 55.5, 'high': 150.4, 'I_low': 151, 'I_high': 200},
    {'low': 150.5, 'high': 250.4, 'I_low': 201, 'I_high': 300},
    {'low': 250.5, 'high': 500.4, 'I_low': 301, 'I_high': 500},
]

breakpoints_pm10 = [
    {'low': 0, 'high': 54, 'I_low': 0, 'I_high': 50},
    {'low': 55, 'high': 154, 'I_low': 51, 'I_high': 100},
    {'low': 155, 'high': 254, 'I_low': 101, 'I_high': 150},
    {'low': 255, 'high': 354, 'I_low': 151, 'I_high': 200},
    {'low': 355, 'high': 424, 'I_low': 201, 'I_high': 300},
    {'low': 425, 'high': 604, 'I_low': 301, 'I_high': 500},
]

breakpoints_so2 = [
    {'low': 0.0, 'high': 35.0, 'I_low': 0, 'I_high': 50},
    {'low': 36.0, 'high': 75.0, 'I_low': 51, 'I_high': 100},
    {'low': 76.0, 'high': 185.0, 'I_low': 101, 'I_high': 150},
    {'low': 186.0, 'high': 304.0, 'I_low': 151, 'I_high': 200},
    {'low': 305.0, 'high': 604.0, 'I_low': 201, 'I_high': 300},
    {'low': 605.0, 'high': 1004.0, 'I_low': 301, 'I_high': 500},
]

breakpoints_no2 = [
    {'low': 0.0, 'high': 53.0, 'I_low': 0, 'I_high': 50},
    {'low': 54.0, 'high': 100.0, 'I_low': 51, 'I_high': 100},
    {'low': 101.0, 'high': 360.0, 'I_low': 101, 'I_high': 150},
    {'low': 361.0, 'high': 649.0, 'I_low': 151, 'I_high': 200},
    {'low': 650.0, 'high': 1249.0, 'I_low': 201, 'I_high': 300},
    {'low': 1250.0, 'high': 2049.0, 'I_low': 301, 'I_high': 500},
]


# Calculate AQI for each hourly prediction
hourly_predictions_df['AQI_PM2.5'] = hourly_predictions_df['Value PM2.5'].apply(calculate_aqi, breakpoints=breakpoints_pm25)
hourly_predictions_df['AQI_PM10'] = hourly_predictions_df['Value PM10'].apply(calculate_aqi, breakpoints=breakpoints_pm10)
hourly_predictions_df['AQI_SO2'] = hourly_predictions_df['Value SO2'].apply(calculate_aqi, breakpoints=breakpoints_so2)
hourly_predictions_df['AQI_NO2'] = hourly_predictions_df['Value NO2'].apply(calculate_aqi, breakpoints=breakpoints_no2)

# Calculate overall AQI as the maximum of the individual AQIs for hourly data
hourly_predictions_df['AQI'] = hourly_predictions_df[['AQI_PM2.5', 'AQI_PM10', 'AQI_SO2', 'AQI_NO2']].max(axis=1)


# Calculate AQI for each daily prediction
daily_predictions_df['AQI_PM2.5'] = daily_predictions_df['Value PM2.5'].apply(calculate_aqi, breakpoints=breakpoints_pm25)
daily_predictions_df['AQI_PM10'] = daily_predictions_df['Value PM10'].apply(calculate_aqi, breakpoints=breakpoints_pm10)
daily_predictions_df['AQI_SO2'] = daily_predictions_df['Value SO2'].apply(calculate_aqi, breakpoints=breakpoints_so2)
daily_predictions_df['AQI_NO2'] = daily_predictions_df['Value NO2'].apply(calculate_aqi, breakpoints=breakpoints_no2)

# Calculate overall AQI as the maximum of the individual AQIs for daily data
daily_predictions_df['AQI'] = daily_predictions_df[['AQI_PM2.5', 'AQI_PM10', 'AQI_SO2', 'AQI_NO2']].max(axis=1)


# Function to determine AQI category
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif 51 <= aqi <= 100:
        return 'Moderate'
    elif 101 <= aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif 151 <= aqi <= 200:
        return 'Unhealthy'
    elif 201 <= aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


# Add AQI category to the DataFrame for hourly data
hourly_predictions_df['AQI_Category'] = hourly_predictions_df['AQI'].apply(categorize_aqi)

# Add AQI category to the DataFrame for daily data
daily_predictions_df['AQI_Category'] = daily_predictions_df['AQI'].apply(categorize_aqi)

# Define holidays
holidays = pd.to_datetime([
    '2024-01-01', '2024-12-25', '2024-12-26',  # Example holidays (New Year, Christmas)
    # Add more holidays here
])


# Adjust AQI for Sundays and holidays for hourly data
def adjust_aqi_for_special_days(row):
    if row.name.weekday() == 6 or row.name in holidays:  # Sunday is represented by 6
        return max(0, row['AQI'] * 0.95)  # Reduce AQI by 5%, ensuring it doesn't go below 0
    return row['AQI']


hourly_predictions_df['Adjusted_AQI'] = hourly_predictions_df.apply(adjust_aqi_for_special_days, axis=1)
hourly_predictions_df['Adjusted_AQI_Category'] = hourly_predictions_df['Adjusted_AQI'].apply(categorize_aqi)

# Adjust AQI for Sundays and holidays for daily data
daily_predictions_df['Adjusted_AQI'] = daily_predictions_df.apply(adjust_aqi_for_special_days, axis=1)
daily_predictions_df['Adjusted_AQI_Category'] = daily_predictions_df['Adjusted_AQI'].apply(categorize_aqi)

# Print the predictions for the next 48 hours (hourly) and next 10 days (daily)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Disable line wrapping

print("\nHourly Predictions for the Next 48 Hours:")
print(hourly_predictions_df[['Value SO2', 'Value PM10', 'Value NO2', 'Value PM2.5', 'Adjusted_AQI', 'Adjusted_AQI_Category']])
print("\nDaily Predictions for the Next 10 Days:")
print(daily_predictions_df[['Value SO2', 'Value PM10', 'Value NO2', 'Value PM2.5', 'Adjusted_AQI', 'Adjusted_AQI_Category']])


print("\nMAE and MSE for each pollutant (Hourly)")
# Calculate MAE and MSE for each pollutant (Hourly)
for pollutant in pollutants:
    actual_data_hourly = hourly_data[pollutant][-forecast_steps_hourly:]  # Select last 'forecast_steps_hourly' hours from actual data
    predicted_data_hourly = hourly_predictions_df[pollutant]  # Match the length of actual data
    mae_hourly = mean_absolute_error(actual_data_hourly, predicted_data_hourly)
    mse_hourly = mean_squared_error(actual_data_hourly, predicted_data_hourly)
    print(f'{pollutant} (Hourly) - MAE: {mae_hourly}, MSE: {mse_hourly}')


print("\nMAE and MSE for each pollutant (Daily)")
# Calculate MAE and MSE for each pollutant (Daily)
for pollutant in pollutants:
    actual_data_daily = daily_data[pollutant][-forecast_steps_daily:]  # Select last 'forecast_steps_daily' days from actual data
    predicted_data_daily = daily_predictions_df[pollutant]  # Match the length of actual data
    mae_daily = mean_absolute_error(actual_data_daily, predicted_data_daily)
    mse_daily = mean_squared_error(actual_data_daily, predicted_data_daily)
    print(f'{pollutant} (Daily) - MAE: {mae_daily}, MSE: {mse_daily}')

# Plot the hourly AQI predictions
plt.figure(figsize=(10, 6))
plt.plot(hourly_predictions_df.index, hourly_predictions_df['Adjusted_AQI'], label='Adjusted AQI (Hourly)', color='blue', marker='o')
plt.title('Adjusted AQI for the Next 48 Hours')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()

# Plot the daily AQI predictions
plt.figure(figsize=(10, 6))
plt.plot(daily_predictions_df.index, daily_predictions_df['Adjusted_AQI'], label='Adjusted AQI (Daily)', color='blue', marker='o')
plt.title('Adjusted AQI for the Next 10 Days')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.legend()
plt.show()


# Plot the predictions against the actual values for each pollutant (Hourly)
for pollutant in pollutants:
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_data.index, hourly_data[pollutant], label='Actual ' + pollutant)
    plt.plot(hourly_predictions_df.index, hourly_predictions_df[pollutant], label='Predicted ' + pollutant, color='red', marker='o')
    plt.title(f'{pollutant} Predictions vs Actual (Hourly)')
    plt.xlabel('Date')
    plt.ylabel(f'Concentration ({pollutant})')
    plt.legend()
    plt.show()


# Plot the predictions against the actual values for each pollutant (Daily)
for pollutant in pollutants:
    plt.figure(figsize=(10, 6))
    plt.plot(daily_data.index, daily_data[pollutant], label='Actual ' + pollutant)
    plt.plot(daily_predictions_df.index, daily_predictions_df[pollutant], label='Predicted ' + pollutant, color='red', marker='o')
    plt.title(f'{pollutant} Predictions vs Actual (Daily)')
    plt.xlabel('Date')
    plt.ylabel(f'Concentration ({pollutant})')
    plt.legend()
    plt.show()


