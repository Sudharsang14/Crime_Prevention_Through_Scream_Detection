import requests

def get_location_ip():
    try:
        res = requests.get("https://ipinfo.io/json")
        data = res.json()
        if "loc" in data:
            lat, lon = data["loc"].split(",")
            return float(lat), float(lon)
    except:
        pass
    return None, None