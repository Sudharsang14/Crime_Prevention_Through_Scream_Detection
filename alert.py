import requests

def send_telegram_alert(risk_level, lat, lon, bot_token, chat_id):
    maps_link = f"https://maps.google.com/?q={lat},{lon}"
    text = f"⚠️ ALERT: {risk_level} scream detected!\nLocation: {maps_link}"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, data=payload)