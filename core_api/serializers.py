# file: serializers.py

from rest_framework import serializers
from .models import AnalysisRun
from urllib.parse import unquote
import os
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import cv2
import supervision as sv
from scipy.spatial import Delaunay
from ultralytics import YOLOv10
from collections import deque
import uuid

class AnalysisRunSerializer(serializers.ModelSerializer):
    status = serializers.CharField(required=False, read_only=True)
    video_url = serializers.CharField(max_length=5000, required=True)
    local_video_path = serializers.CharField(required=False, read_only=True)
    analyzed_video_url = serializers.CharField(required=False, read_only=True)
    flocking_index = serializers.CharField(required=False, read_only=True)
    avg_tel = serializers.CharField(required=False, read_only=True)
    avg_ta = serializers.CharField(required=False, read_only=True)
    avg_group_speed = serializers.CharField(required=False, read_only=True)

    class Meta:
        model = AnalysisRun
        fields = [
            "id", "run_name", "video_url", "analyzed_video_url",
            "flocking_index", "avg_tel", "avg_ta", "avg_group_speed",
            "timestamp", "status", "local_video_path"
        ]
        read_only_fields = [
            "flocking_index", "avg_tel", "avg_ta", "avg_group_speed",
            "timestamp", "status", "local_video_path", "analyzed_video_url"
        ]

    def create(self, validated_data):
        run_name = validated_data.get("run_name")
        video_url = validated_data.get("video_url")
        run_uuid = uuid.uuid4().hex[:8]  # Unique identifier per run

        local_video_path = self.download_video_from_firebase(video_url, run_uuid)

        if local_video_path is None:
            validated_data["status"] = "Video download failed"
            return validated_data

        try:
            results = self.process_video(local_video_path, run_uuid)
            flocking_index = results["flocking_index"]
            avg_tel = results["avg_tel"]
            avg_ta = results["avg_ta"]
            avg_group_speed = results["avg_group_speed"]
            run = AnalysisRun.objects.create(
                run_name=run_name,
                video_url=video_url,
                analyzed_video_url=results["output_video_path"],
                flocking_index=flocking_index,
                avg_tel=avg_tel,
                avg_ta=avg_ta,
                avg_group_speed=avg_group_speed
            )

            validated_data["output_video_path"] = results["output_video_path"]
            validated_data["local_video_path"] = local_video_path
            validated_data["status"] = "Analysis complete"
            validated_data["flocking_index"] = flocking_index
            validated_data["avg_tel"] = avg_tel
            validated_data["avg_ta"] = avg_ta
            validated_data["avg_group_speed"] = avg_group_speed

            if os.path.exists(local_video_path):
                os.remove(local_video_path)

            if os.path.exists(results["output_local_path"]):
                os.remove(results["output_local_path"])

            return validated_data

        except Exception as e:
            validated_data["status"] = f"Processing failed: {e}"
            return validated_data

    def download_video_from_firebase(self, video_url, run_uuid):
        if not firebase_admin._apps:
            cred = credentials.Certificate("key.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'neon-gist-456416-s8.firebasestorage.app'
            })

        try:
            p = video_url.split("/o/")[1].split("?alt=")[0]
            bpath = unquote(p)
            os.makedirs("downloaded_videos", exist_ok=True)
            f = os.path.basename(bpath)
            name, ext = os.path.splitext(f)
            unique_f = f"{name}_{run_uuid}{ext}"
            path = os.path.join("downloaded_videos", unique_f)
            b = storage.bucket().blob(bpath)
            b.download_to_filename(path)
            return path
        except Exception:
            return None

    def process_video(self, input_video_path: str, run_uuid: str):
        model = YOLOv10('./best.pt')
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        trace_annotator = sv.TraceAnnotator()

        frame_rate = 25
        fish_tracks = {}
        fish_speed_buffers = {}
        buffer_size = 10
        flocking_index_values = []
        average_speeds = []
        average_triangle_edge_lengths = []
        average_triangle_areas = []

        def calculate_speed(trajectory, frame_rate):
            speeds = []
            for i in range(1, len(trajectory)):
                prev = trajectory[i-1]
                curr = trajectory[i]
                distance = np.linalg.norm(np.array(curr) - np.array(prev))
                speed = distance * frame_rate
                speeds.append(speed)
            return speeds

        def calculate_average_speed(speed_buffers):
            all_speeds = []
            for speeds in speed_buffers.values():
                all_speeds.extend(speeds)
            return sum(all_speeds) / len(all_speeds) if all_speeds else 0

        def calculate_flocking_index(points, delaunay_tri, num_fishes=5):
            flocking_index = 0
            for simplex in delaunay_tri.simplices:
                edge_lengths = [
                    np.linalg.norm(points[simplex[i]] - points[simplex[(i + 1) % 3]]) for i in range(3)
                ]
                inverse_edge_sum = sum(edge_lengths)/1000
                neighborhood_density = len(simplex) / inverse_edge_sum
                flocking_index += neighborhood_density
            normalized = flocking_index * (num_fishes / len(points))
            return normalized * 5

        def calculate_triangle_area(points):
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            return abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2

        def calculate_average_triangle_area(points, delaunay_tri):
            total_area, count = 0, 0
            for simplex in delaunay_tri.simplices:
                total_area += calculate_triangle_area(points[simplex])
                count += 1
            return total_area / count if count else 0

        def calculate_average_triangle_edge_length(points, delaunay_tri, speeds):
            total, count = 0, 0
            for simplex in delaunay_tri.simplices:
                for i in range(3):
                    total += np.linalg.norm(points[simplex[i]] - points[simplex[(i + 1) % 3]])
                    count += 1
            return (total / count) * np.mean(speeds) if count else 0

        def draw_delaunay(frame, centroids):
            points = np.array(centroids)
            if len(points) >= 3:
                delaunay_tri = Delaunay(points)
                flocking = calculate_flocking_index(points, delaunay_tri)
                flocking_index_values.append(flocking)
                speeds = [s for t in fish_tracks.values() for s in calculate_speed(t, frame_rate)]
                average_speeds.append(np.mean(speeds))
                average_triangle_areas.append(calculate_average_triangle_area(points, delaunay_tri))
                average_triangle_edge_lengths.append(calculate_average_triangle_edge_length(points, delaunay_tri, speeds))
                for simplex in delaunay_tri.simplices:
                    pts = points[simplex]
                    cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            return frame, len(points)

        def upload_video_to_firebase(local_path, run_uuid):
            if not firebase_admin._apps:
                cred = credentials.Certificate("key.json")
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'neon-gist-456416-s8.appspot.com'
                })
            filename = os.path.basename(local_path)
            firebase_path = f"processed_videos/{run_uuid}_{filename}"
            blob = storage.bucket().blob(firebase_path)
            blob.upload_from_filename(local_path, content_type="video/mp4")
            blob.make_public()
            return blob.public_url

        def callback(frame: np.ndarray, frame_number: int) -> np.ndarray:
            results = model(frame)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(results))
            labels = []
            for class_id, tracker_id, bbox in zip(detections.class_id, detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = bbox
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                fish_tracks.setdefault(tracker_id, deque(maxlen=buffer_size)).append(centroid)
                speeds = calculate_speed(fish_tracks[tracker_id], frame_rate)
                fish_speed_buffers.setdefault(tracker_id, deque(maxlen=buffer_size)).extend(speeds)
                avg_speed = calculate_average_speed(fish_speed_buffers)
                labels.append(f"#{tracker_id} {results.names[class_id]} Avg Speed: {avg_speed:.2f} px/s")
            annotated = label_annotator.annotate(box_annotator.annotate(frame.copy(), detections), detections, labels)
            centroids = [(int((x1+x2)/2), int((y1+y2)/2)) for x1,y1,x2,y2 in detections.xyxy]
            annotated, _ = draw_delaunay(annotated, centroids)
            return trace_annotator.annotate(annotated, detections)

        os.makedirs('./analyzed_video', exist_ok=True)
        input_name = os.path.splitext(os.path.basename(input_video_path))[0]
        output_path = f"./analyzed_video/{input_name}_{run_uuid}_analyzed_output.mp4"

        sv.process_video(
            source_path=input_video_path,
            target_path=output_path,
            callback=callback
        )

        return {
            "flocking_index": f"{np.nanmean(flocking_index_values):.2f}",
            "avg_group_speed": f"{np.nanmean(average_speeds):.2f}",
            "avg_ta": f"{np.nanmean(average_triangle_areas)/100:.2f} px²",
            "avg_tel": f"{np.nanmean(average_triangle_edge_lengths)/1000:.2f} px",
            "output_video_path": upload_video_to_firebase(output_path, run_uuid),
            "output_local_path": output_path
        }
