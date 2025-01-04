from django.shortcuts import render
from django.http import HttpResponse

import requests #This library helps us fetch data from API
import pandas as pd #for handling and analyzing data
import numpy as np #for numerical operations
import joblib
import pytz
import os
from sklearn.model_selection import train_test_split #to split data intro training and testing sets
from sklearn.preprocessing import LabelEncoder #to convert categorical data into numerical
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor #models for classification and regression tasks
from sklearn.metrics import mean_squared_error #to measure the accuracy of our predictions
from datetime import datetime, timedelta #to handle date and time
from geopy.geocoders import Nominatim
from forecast import MODEL_DIR
from forecast.api_key import API_KEY
from timezonefinder import TimezoneFinder

BASE_URL = 'https://api.openweathermap.org/data/2.5/' #base url for API requests

# Helper function to get rounded value or NaN
def get_rounded_value(data, key):
    value = data.get(key, np.nan)
    return round(value) if not np.isnan(value) else np.nan

# Get user's location based on IP address
def get_user_location(ip_address):
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json")
        data = response.json()
        loc = data.get("loc", None)
        if loc:
            lat, lon = loc.split(",")
            geolocator = Nominatim(user_agent="weather_app")
            location = geolocator.reverse(f"{lat}, {lon}")
            address = location.raw['address']
            city = address.get('city')
            country = address.get('country')
            if city and country:
                return f"{city}, {country}"
    except Exception as e:
        print(f"Error getting user location: {e}")
    return "Sofia, Bulgaria"

# Function to get the timezone of a city
def get_timezone(city):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city)
    if location:
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=location.longitude, lat=location.latitude)
        return timezone_str
    return None

# 1. Fetch Current Weather Data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return {'error': True, 'code': data.get('cod'), 'message': "OpenWeather: " + data.get('message')}

    return_data = {}
    return_data['city'] = data.get('name', np.nan)

    main_data = data.get('main', {})
    return_data['current_temp'] = get_rounded_value(main_data, 'temp')
    return_data['feels_like'] = get_rounded_value(main_data, 'feels_like')
    return_data['temp_min'] = get_rounded_value(main_data, 'temp_min')
    return_data['temp_max'] = get_rounded_value(main_data, 'temp_max')
    return_data['humidity'] = get_rounded_value(main_data, 'humidity')
    return_data['pressure'] = get_rounded_value(main_data, 'pressure')

    weather_data = data.get('weather', [{}])
    return_data['description'] = weather_data[0].get('description', np.nan) if weather_data else np.nan

    sys_data = data.get('sys', {})
    return_data['country'] = sys_data.get('country', np.nan)

    wind_data = data.get('wind', {})
    return_data['wind_gust_dir'] = wind_data.get('deg', np.nan)
    return_data['Wind_Gust_Speed'] = wind_data.get('speed', np.nan)

    clouds_data = data.get('clouds', {})
    return_data['clouds'] = clouds_data.get('all', np.nan)

    return_data['visibility'] = data.get('visibility', np.nan)

    return return_data

#2. Read Historical Data
def read_historical_data(filename):
  df = pd.read_csv(filename)
  df = df.dropna() #remove rows with missing values
  df = df.drop_duplicates()
  return df

#3. Prepare data for training
def prepare_data(data):
  le = LabelEncoder()
  data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

  #define the feature variable and target variable
  X = data[['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']] #feature variables
  y = data['RainTomorrow'] #target variable

  return X, y, le

#4. Train Rain Prediction Model
def train_rain_model(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train) #train the model

  y_pred = model.predict(X_test) #to make prediction on the test set
  print("Mean Squared Error for Rain Model")
  print(mean_squared_error(y_test, y_pred))

  return model

#5. Prepare regression data
def prepare_regression_data(data, feature):
  X, y = [], [] #initialize list for feature and target values

  for i in range(len(data) - 1): #each feature from X is matched with the correct output from y
    X.append(data[feature].iloc[i]) #X gets the feature values
    y.append(data[feature].iloc[i+1]) #y gets the next feature values
                                      #Regressions will learn the relation between past and future values

  X = np.array(X).reshape(-1, 1) #2d array with 1 column for regression model
  y = np.array(y) #target values
  return X, y

#6. Train Regression Model
def train_regression_model(X, y):
  model = RandomForestRegressor(n_estimators=100, random_state=42) #Use 100 decision trees and same random state for reproduceability
  model.fit(X, y)
  return model

#7. Predict Future
def predict_future(model, current_value):
  predictions = [current_value]

  for i in range(5): #predict the next 5 time steps
    next_value = model.predict(np.array([[predictions[-1]]]))
    predictions.append(next_value[0])

  return predictions[1:]

# Function to prepare and train the rain prediction model
def prepare_and_train_rain_model(historical_data):
    model_path = os.path.join(MODEL_DIR, 'rain_model.pkl')
    le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    if os.path.exists(model_path) and os.path.exists(le_path):
        rain_model = joblib.load(model_path)
        le = joblib.load(le_path)
    else:
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)
        joblib.dump(rain_model, model_path)
        joblib.dump(le, le_path)
    return rain_model

# Function to prepare and train the regression model
def prepare_and_train_regression_model(historical_data, feature):
    model_path = os.path.join(MODEL_DIR, f'{feature}_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        X, y = prepare_regression_data(historical_data, feature)
        model = train_regression_model(X, y)
        joblib.dump(model, model_path)
    return model

#8. Weather Analysis Function
def weather_view(request):
  if request.method == 'POST':
    city = request.POST.get('city')
  else:
    ip_address = request.META.get('REMOTE_ADDR', None)
    city = get_user_location(ip_address)
    
  current_weather = get_current_weather(city)

  if 'error' in current_weather:
    return render(request, 'weather.html', {'error': current_weather['message'], 'code': current_weather['code'], 'description': 'error'})

  # Get the timezone of the city
  timezone_str = get_timezone(city)
  if timezone_str:
      timezone = pytz.timezone(timezone_str)
      current_time = datetime.now(timezone)
  else:
      current_time = datetime.now()

  #load historical data
  csv_path = os.path.join('/Users/i554234/repos/WeatherPredictorApp/weatherPredictor/train_data/averageBulgarianWeather.csv')
  historical_data = read_historical_data(csv_path)

  # Prepare and train rain prediction model
  rain_model = prepare_and_train_rain_model(historical_data)

  # Load the LabelEncoder
  le_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
  if os.path.exists(le_path):
      le = joblib.load(le_path)
  else:
      _, _, le = prepare_data(historical_data)
      joblib.dump(le, le_path)

  #map wind direction to compass points
  wind_deg = current_weather['wind_gust_dir'] % 360
  compass_points = [
      ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
      ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
      ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
      ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
      ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
      ("NNW", 326.25, 348.75), ("N", 348.75, 360.01)
  ]
  compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
  compass_direction_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

  current_data = {
      'MinTemp': current_weather['temp_min'],
      'MaxTemp': current_weather['temp_max'],
      'WindGustSpeed': current_weather['Wind_Gust_Speed'],
      'Humidity': current_weather['humidity'],
      'Pressure': current_weather['pressure'],
      'Temp': current_weather['current_temp']
  }

  current_df = pd.DataFrame([current_data])

  #rain prediction
  rain_prediction = rain_model.predict(current_df)[0]

  # Prepare and train regression models for temperature and humidity
  temp_model = prepare_and_train_regression_model(historical_data, 'Temp')
  hum_model = prepare_and_train_regression_model(historical_data, 'Humidity')

  #predict future temperature and humidity
  future_temp = predict_future(temp_model, current_weather['temp_min'])
  future_humidity = predict_future(hum_model, current_weather['humidity'])

  #prepare time for future predictions
  future_times = [(current_time + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
  
  #store each value separately
  time1, time2, time3, time4, time5 = future_times
  temp1, temp2, temp3, temp4, temp5 = future_temp
  hum1, hum2, hum3, hum4, hum5 = future_humidity

  # Pass data to template
  context = {
      'location': city,
      'city': current_weather['city'],
      'country': current_weather['country'],
      'current_temp': current_weather['current_temp'],
      'feels_like': current_weather['feels_like'],
      'MinTemp': current_weather['temp_min'],
      'MaxTemp': current_weather['temp_max'],
      'humidity': current_weather['humidity'],
      'description': current_weather['description'],
      'clouds': current_weather['clouds'],
      'visibility': current_weather['visibility'],
      'time': current_time.strftime("%H:%M"),
      'date': current_time.strftime("%B %d, %Y"),
      'wind_gust_dir': compass_direction,
      'pressure': current_weather['pressure'],
      'wind': current_weather['Wind_Gust_Speed'],
      'rain_prediction': 'Yes' if rain_prediction == 1 else 'No',
      'time1': time1, 'time2': time2, 'time3': time3, 'time4': time4, 'time5': time5,
      'temp1': f"{round(temp1, 1)}", 'temp2': f"{round(temp2, 1)}", 'temp3': f"{round(temp3, 1)}", 'temp4': f"{round(temp4, 1)}", 'temp5': f"{round(temp5, 1)}",
      'hum1': f"{round(hum1, 1)}", 'hum2': f"{round(hum2, 1)}", 'hum3': f"{round(hum3, 1)}", 'hum4': f"{round(hum4, 1)}", 'hum5': f"{round(hum5, 1)}",
  }
  return render(request, 'weather.html', context)