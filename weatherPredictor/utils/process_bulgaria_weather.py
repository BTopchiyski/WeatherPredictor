import json
import csv
import os

# Define the paths to the JSON files and the output CSV file
json_files = [
    os.path.join(os.path.dirname(__file__), '..', 'train_data', 'plovdivWeather.json'),
    os.path.join(os.path.dirname(__file__), '..', 'train_data', 'sofiaWeather.json'),
    os.path.join(os.path.dirname(__file__), '..', 'train_data', 'burgasWeather.json')
]
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'train_data', 'averageBulgarianWeather.csv')

# Define the threshold for determining if it will rain tomorrow
PRECIPITATION_THRESHOLD = 0.9  # Example threshold value in mm

# Function to read and process JSON data
def read_weather_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    weather_data = []
    for entry in data['result']:
        max_precipitation = entry['precipitation']['max']
        rain_tomorrow = 'Yes' if max_precipitation > PRECIPITATION_THRESHOLD else 'No'
        
        # Convert temperatures from Kelvin to Celsius and round to one decimal place
        min_temp_celsius = round(entry['temp']['average_min'] - 273.15, 1)
        max_temp_celsius = round(entry['temp']['average_max'] - 273.15, 1)
        temp_celsius = round(entry['temp']['mean'] - 273.15, 1)
        
        day_data = {
            'Month': entry['month'],
            'Day': entry['day'],
            'MinTemp': min_temp_celsius,
            'MaxTemp': max_temp_celsius,
            'WindGustSpeed': entry['wind']['max'],
            'Humidity': entry['humidity']['mean'],
            'Pressure': entry['pressure']['mean'],
            'Temp': temp_celsius,
            'RainTomorrow': rain_tomorrow
        }
        weather_data.append(day_data)
    
    return weather_data

# Read and process data from all JSON files
all_weather_data = [read_weather_data(json_file) for json_file in json_files]

# Calculate the average values for each field across the three cities
average_weather_data = []
for day_entries in zip(*all_weather_data):
    avg_day_data = {
        'MinTemp': round(sum(entry['MinTemp'] for entry in day_entries) / len(day_entries), 1),
        'MaxTemp': round(sum(entry['MaxTemp'] for entry in day_entries) / len(day_entries), 1),
        'WindGustSpeed': round(sum(entry['WindGustSpeed'] for entry in day_entries) / len(day_entries), 1),
        'Humidity': round(sum(entry['Humidity'] for entry in day_entries) / len(day_entries), 1),
        'Pressure': round(sum(entry['Pressure'] for entry in day_entries) / len(day_entries), 1),
        'Temp': round(sum(entry['Temp'] for entry in day_entries) / len(day_entries), 1),
        'RainTomorrow': 'Yes' if any(entry['RainTomorrow'] == 'Yes' for entry in day_entries) else 'No'
    }
    average_weather_data.append(avg_day_data)

# Write the averaged data to a new CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['MinTemp', 'MaxTemp', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp', 'RainTomorrow']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in average_weather_data:
        writer.writerow(row)

print("Data has been successfully processed and written to averageBulgarianWeather.csv")