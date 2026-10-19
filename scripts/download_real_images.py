import urllib.request
import json
import os
from tqdm import tqdm

os.makedirs('data/raw', exist_ok=True)
print("Fetching real image list...")
url = 'https://picsum.photos/v2/list?page=1&limit=50'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

print(f"Downloading {len(data)} real images...")
for item in tqdm(data):
    # Resize to 500x500 for faster download and processing
    img_url = f"https://picsum.photos/id/{item['id']}/500/500"
    img_path = f"data/raw/picsum_{item['id']}.jpg"
    if not os.path.exists(img_path):
        urllib.request.urlretrieve(img_url, img_path)

print("Real images downloaded successfully!")
