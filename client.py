import requests

response = requests.post(
    "http://127.0.0.1:8000/analyze",
    json={
        "temperature": 34,
        "rain": 0,
        "weekend": 1,
        "event": 1
    }
)

print(response.json())