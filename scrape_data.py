import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Define directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

BASE_URL = "https://transcripts.recursiveprojects.cloud"
ARCHIVE_URL = f"{BASE_URL}/archive"
HEADERS = {"User-Agent": "F1-NLP-Course-Project/1.0 (+https://github.com/wraith/f1-nlp-project)"}

def get_sessions():
    print(f"Fetching archive page: {ARCHIVE_URL}")
    res = requests.get(ARCHIVE_URL, headers=HEADERS)
    if res.status_code != 200:
        print("Failed to fetch archive page")
        return []
    
    soup = BeautifulSoup(res.text, "html.parser")
    sessions = []
    # Find all session links in the sidebar
    links = soup.find_all("a", href=lambda x: x and x.startswith("/archive/"))
    for link in links:
        href = link.get("href")
        name = link.get_text(strip=True)
        # We only want main Grand Prix sessions (Races) to keep data high-quality and size reasonable
        # Main races typically end with "Grand Prix" and do not contain Qualifying, Practice, or Sprint
        if name.endswith("Grand Prix") and not any(k in name for k in ["Qualifying", "Practice", "Sprint"]):
            sessions.append({
                "name": name,
                "url": f"{BASE_URL}{href}"
            })
    return sessions

def get_drivers(session_url):
    print(f"Fetching drivers for session: {session_url}")
    res = requests.get(session_url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to fetch session page: {session_url}")
        return []
    
    soup = BeautifulSoup(res.text, "html.parser")
    drivers = []
    # Find all driver links (which look like /archive/{session_id}/{driver_id})
    links = soup.find_all("a", href=lambda x: x and x.startswith(session_url.replace(BASE_URL, "")))
    for link in links:
        href = link.get("href")
        name = link.get_text(strip=True)
        # Ensure it's a driver URL (should have two slashes after /archive/)
        parts = href.strip("/").split("/")
        if len(parts) == 3:  # archive, session_id, driver_id
            drivers.append({
                "name": name,
                "url": f"{BASE_URL}{href}"
            })
    return drivers

def get_messages(driver_url, gp_name, driver_name):
    print(f"Fetching messages for {driver_name} at {gp_name}...")
    res = requests.get(driver_url, headers=HEADERS)
    if res.status_code != 200:
        print(f"Failed to fetch driver page: {driver_url}")
        return []
    
    soup = BeautifulSoup(res.text, "html.parser")
    messages = []
    
    # Message container divs have class dark:bg-zinc-800
    message_divs = soup.find_all("div", class_=lambda x: x and "dark:bg-zinc-800" in x)
    for div in message_divs:
        ts_span = div.find("span", class_="timestamp")
        timestamp = ts_span.get("data-timestamp") if ts_span else None
        grow_div = div.find(class_="grow")
        message_text = grow_div.get_text(strip=True) if grow_div else None
        
        if message_text:
            messages.append({
                "grand_prix": gp_name,
                "session_type": "Race",
                "driver_name": driver_name,
                "timestamp": timestamp,
                "message_text": message_text
            })
    return messages

def main():
    sessions = get_sessions()
    print(f"Found {len(sessions)} Grand Prix race sessions.")
    
    # Scrape the first 3 sessions to keep execution time fast and data clean (around 3000-5000 messages)
    target_sessions = sessions[:3]
    all_data = []
    
    for sess in target_sessions:
        print(f"\n--- Scraping {sess['name']} ---")
        drivers = get_drivers(sess["url"])
        print(f"Found {len(drivers)} drivers in this session.")
        
        for drv in drivers:
            # Respectful crawling: delay to avoid hammering the server
            time.sleep(1.0)
            msgs = get_messages(drv["url"], sess["name"], drv["name"])
            print(f"  Scraped {len(msgs)} messages.")
            all_data.extend(msgs)
            
            # Temporary checkpoint save
            if len(all_data) % 200 == 0 or len(all_data) < 500:
                df_temp = pd.DataFrame(all_data)
                df_temp.to_csv(os.path.join(PROCESSED_DIR, "f1_radio_corpus.csv"), index=False)
                
    df = pd.DataFrame(all_data)
    csv_path = os.path.join(PROCESSED_DIR, "f1_radio_corpus.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nScraping complete! Saved {len(df)} total messages to {csv_path}")

if __name__ == "__main__":
    main()
