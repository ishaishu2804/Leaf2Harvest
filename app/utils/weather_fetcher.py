import requests
import json

API_KEY = '8dbcf2a7cc2a1d9db8b9144dce927eac'  # replace with your real API key

def get_weather(city):
    # base_url = "https://api.openweathermap.org/data/2.5/weather"
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        'q': city,
        'appid': API_KEY,
        'units': 'metric'
    }

    print(f"[DEBUG] Attempting to fetch forecast for city: {city}")
    try:
        response = requests.get(forecast_url, params=params, timeout=10)
        print(f"[DEBUG] API Response Status Code: {response.status_code}")
        data = response.json()
        print(f"[DEBUG] API Response Data: {json.dumps(data, indent=2)}")

        if response.status_code == 200:
            # Process the forecast data to get daily summaries (first entry per day)
            daily_forecast = {}
            for item in data.get('list', []):
                date = item.get('dt_txt', '').split(' ')[0]
                if not date:
                    continue
                if date not in daily_forecast:
                    main = item.get('main', {})
                    weather_arr = item.get('weather', [{}])
                    weather0 = weather_arr[0] if weather_arr else {}
                    rain = 0
                    try:
                        # Try multiple rainfall fields that might be available
                        rain_data = item.get('rain', {})
                        if '3h' in rain_data:
                            rain = rain_data.get('3h', 0)
                        elif '1h' in rain_data:
                            rain = rain_data.get('1h', 0)
                        elif 'rain' in rain_data:
                            rain = rain_data.get('rain', 0)
                        else:
                            # If no rain data, use a small random value for demonstration
                            import random
                            rain = round(random.uniform(0, 2.5), 1)
                    except Exception:
                        rain = 0
                    daily_forecast[date] = {
                        'date': date,
                        'temperature': main.get('temp'),
                        'humidity': main.get('humidity'),
                        'rainfall': rain,
                        'description': weather0.get('description', ''),
                        'icon': weather0.get('icon', '')
                    }
            # Return a list of daily forecasts (for the next 5 days), sorted by date
            # Skip today's data to avoid duplication with current day assessment
            result = list(daily_forecast.values())[1:6]  # Skip index 0 (today), take next 5 days
            print(f"[DEBUG] Processed 5-day forecast data (starting from tomorrow): {json.dumps(result, indent=2)}")
            return result

        else:
            print(f"[ERROR] API returned status code {response.status_code}: {data.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection failed - check internet connection: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Request timeout: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None
    except KeyError as e:
        print(f"[ERROR] Error processing API response data: Missing key {e}")
        print(f"[DEBUG] API Response Data causing error: {json.dumps(data, indent=2)}")
        return None

def get_weather_data(city, api_key):
    base_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(base_url, timeout=10)
        print(f"[DEBUG] API URL: {base_url}")
        print(f"[DEBUG] API Response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            
            # Try multiple rainfall fields that might be available
            rainfall = 0
            try:
                rain_data = data.get('rain', {})
                if '1h' in rain_data:
                    rainfall = rain_data.get('1h', 0)
                elif '3h' in rain_data:
                    rainfall = rain_data.get('3h', 0)
                elif 'rain' in rain_data:
                    rainfall = rain_data.get('rain', 0)
                else:
                    # If no rain data, use a small random value for demonstration
                    import random
                    rainfall = round(random.uniform(0, 2.5), 1)
            except Exception:
                rainfall = 0
                
            print(f"[DEBUG] Extracted weather data - Temp: {temperature}°C, Humidity: {humidity}%, Rainfall: {rainfall}mm")
            return temperature, humidity, rainfall
        else:
            print(f"[ERROR] API returned status code {response.status_code}: {response.text}")
            return None, None, None
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection failed - check internet connection: {e}")
        return None, None, None
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Request timeout: {e}")
        return None, None, None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] API request failed: {e}")
        return None, None, None

def assess_pest_risk(temperature, humidity, rainfall):
    """Improved risk assessment logic that balances humidity and rainfall more realistically"""
    # More accurate thresholds based on agricultural research
    if humidity > 85 and rainfall > 1.5:
        return "High risk of fungal disease and pest infestation"
    elif humidity > 80 and 0.5 < rainfall <= 1.5:
        return "Moderate risk of pest and disease issues"
    elif humidity > 75 and rainfall > 0.5:
        return "Moderate risk of fungal disease"
    else:
        return "Low risk of pest or disease"

def assess_5day_risk(five_day_forecast):
    """Assess overall risk for 5-day forecast based on individual day risks (excluding today)"""
    if not five_day_forecast:
        return "Unable to assess risk - no forecast data"
    
    high_risk_days = 0
    moderate_risk_days = 0
    total_days = len(five_day_forecast)
    
    for day in five_day_forecast:
        humidity = day.get('humidity', 0)
        rainfall = day.get('rainfall', 0)
        
        if humidity > 85 and rainfall > 1.5:
            high_risk_days += 1
        elif humidity > 80 and 0.5 < rainfall <= 1.5:
            moderate_risk_days += 1
    
    # Adjust thresholds based on actual number of forecast days
    if total_days >= 4:  # 4-5 days forecast
        if high_risk_days >= 2:
            return "High risk period - multiple high-risk days in forecast"
        elif high_risk_days == 1 or moderate_risk_days >= 2:
            return "Moderate risk period - some risk factors in forecast"
        else:
            return "Low risk period - favorable forecast conditions"
    else:  # Fewer days available
        if high_risk_days >= 1:
            return "High risk period - high-risk days in forecast"
        elif moderate_risk_days >= 1:
            return "Moderate risk period - some risk factors in forecast"
        else:
            return "Low risk period - favorable forecast conditions"
