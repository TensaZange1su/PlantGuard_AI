import random

def simulate_sensor_data():
    return {
        "temperature": round(random.uniform(18, 35), 2),
        "humidity": round(random.uniform(40, 85), 2),
        "soil_moisture": round(random.uniform(30, 90), 2)
    }
