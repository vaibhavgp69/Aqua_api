from google.cloud import storage
import os


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"

def upload_video(local_path, destination_blob_name, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(local_path)
    print(f"✅ Video uploaded to: gs://{bucket_name}/{destination_blob_name}")

if __name__ == "__main__":
    local_video_path = "analyzed_video\su06jun_tankb_nofedt - Trim_analyzed_output_streamable.mp4"
    bucket_name = "neon-gist-456416-s8.firebasestorage.app"
    destination_path = "processed_videos/my_video.mp4"

    upload_video(local_video_path, destination_path, bucket_name)