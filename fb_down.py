import firebase_admin
from firebase_admin import credentials, storage
from urllib.parse import unquote
import os

def download_video_from_firebase(video_url):
    if not firebase_admin._apps:
        cred = credentials.Certificate("key.json")
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'neon-gist-456416-s8.firebasestorage.app'
        })

    try:
        print(f"Video URL: {video_url}")  # Debug: Show the URL being processed

        # Extract the file path after the bucket name (skip the domain part)
        path = video_url.split('.app/')[1]
        bpath = unquote(path)  # Decode URL encoding like %20 for spaces

        print(f"Extracted Path: {bpath}")  # Debug: Show the extracted path

        # Prepare local download path
        os.makedirs("downloaded_videos", exist_ok=True)
        f = os.path.basename(bpath)
        download_path = os.path.join("downloaded_videos", f)

        # Download the file from Firebase Storage
        b = storage.bucket().blob(bpath)
        b.download_to_filename(download_path)
        return download_path
    except Exception as e:
        print(f"Error: {e}")
        return None

# Test with your Firebase URL
download_video_from_firebase('https://storage.googleapis.com/neon-gist-456416-s8.firebasestorage.app/processed_videos/su06jun_tankb_nofedt%20-%20Trim_analyzed_output.mp4')
