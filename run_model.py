import os
import numpy as np
import cv2
import supervision as sv
from scipy.spatial import Delaunay
from ultralytics import YOLOv10
from collections import deque

def process_video(input_video_path: str):
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
        if len(all_speeds) == 0:
            return 0
        return sum(all_speeds) / len(all_speeds)

    def calculate_flocking_index(points, delaunay_tri, num_fishes=5):
        flocking_index = 0
        for simplex in delaunay_tri.simplices:
            edge_lengths = []
            for i in range(3):
                edge_lengths.append(np.linalg.norm(points[simplex[i]] - points[simplex[(i + 1) % 3]]))
            inverse_edge_sum = sum(length for length in edge_lengths)/1000
            neighborhood_density = len(simplex) / inverse_edge_sum
            flocking_index += neighborhood_density
        normalized_flocking_index = flocking_index * (num_fishes / len(points))
        return normalized_flocking_index * 5

    def calculate_triangle_area(points):
        x1, y1 = points[0]
        x2, y2 = points[1]
        x3, y3 = points[2]
        area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2
        return area

    def calculate_average_triangle_area(points, delaunay_tri):
        total_area = 0
        triangle_count = 0

        for simplex in delaunay_tri.simplices:
            triangle_vertices = points[simplex]
            triangle_area = calculate_triangle_area(triangle_vertices)
            total_area += triangle_area
            triangle_count += 1

        if triangle_count == 0:
            return 0
        average_triangle_area = total_area / triangle_count
        return average_triangle_area

    def calculate_average_triangle_edge_length(points, delaunay_tri, speeds):
        total_edge_length = 0
        edge_count = 0

        for simplex in delaunay_tri.simplices:
            for i in range(3):
                edge_length = np.linalg.norm(points[simplex[i]] - points[simplex[(i + 1) % 3]])
                total_edge_length += edge_length
                edge_count += 1
        if edge_count == 0:
            return 0

        average_triangle_edge_length = total_edge_length / edge_count
        adjusted_edge_length = average_triangle_edge_length * np.mean(speeds)

        return adjusted_edge_length

    def draw_delaunay(frame, centroids):
        points = np.array(centroids)

        if len(points) < 3:
            for centroid in centroids:
                cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    cv2.line(frame, centroids[i], centroids[j], (255, 0, 0), 2)
        else:
            delaunay_tri = Delaunay(points)

            flocking_index = calculate_flocking_index(points, delaunay_tri)
            flocking_index_values.append(flocking_index)
            min_flocking_index = 0
            max_flocking_index = 1000
            normalized_index = (flocking_index - min_flocking_index) / (max_flocking_index - min_flocking_index) * 1000
            flocking_index_values.append(flocking_index)
            flocking_index_values.append(normalized_index)
            flocking_text = f"Flocking Index: {normalized_index:.2f}"

            speeds = []
            for tracker_id, trajectory in fish_tracks.items():
                speeds.extend(calculate_speed(trajectory, frame_rate))

            average_speeds.append(np.mean(speeds))

            avg_triangle_area = calculate_average_triangle_area(points, delaunay_tri)
            avg_edge_length = calculate_average_triangle_edge_length(points, delaunay_tri, speeds)

            average_triangle_areas.append(avg_triangle_area)
            average_triangle_edge_lengths.append(avg_edge_length)

            density_text = f"Avg Triangle Area: {avg_triangle_area/100:.2f}"
            area_text = f"Avg Edge Length: {avg_edge_length/1000:.2f}"
            text_size = cv2.getTextSize(flocking_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]
            box_width = text_size[0] + 20
            box_height = text_size[1] * 3 + 40
            cv2.rectangle(frame, (20, 20), (20 + 480, 20 + 150), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (20 + 480, 20 + 150), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, flocking_text, (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, density_text, (30, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, area_text, (30, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

            for simplex in delaunay_tri.simplices:
                pts = points[simplex]
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        return frame, len(points)


    def callback(frame: np.ndarray, frame_number: int) -> np.ndarray:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        try:
            labels = []
            for class_id, tracker_id, bbox in zip(detections.class_id, detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = bbox
                centroid_x = int((x1 + x2) / 2)
                centroid_y = int((y1 + y2) / 2)
                centroid = (centroid_x, centroid_y)

                if tracker_id not in fish_tracks:
                    fish_tracks[tracker_id] = deque(maxlen=buffer_size)
                fish_tracks[tracker_id].append(centroid)

                trajectory = fish_tracks[tracker_id]
                speeds = calculate_speed(trajectory, frame_rate)

                if tracker_id not in fish_speed_buffers:
                    fish_speed_buffers[tracker_id] = deque(maxlen=buffer_size)

                if speeds:
                    fish_speed_buffers[tracker_id].extend(speeds)
                average_speed = calculate_average_speed(fish_speed_buffers)

                label = f"#{tracker_id} {results.names[class_id]} Avg Speed: {average_speed:.2f} px/s"
                labels.append(label)

            annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
        except ValueError as e:
            print(f"ValueError encountered: {e}")
            return frame

        centroids = []
        for detection in detections.xyxy:
            x1, y1, x2, y2 = detection
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            centroids.append((centroid_x, centroid_y))
        annotated_frame, len_p = draw_delaunay(annotated_frame, centroids)

        return trace_annotator.annotate(annotated_frame, detections=detections)


    output_video_dir = './analyzed_video'
    os.makedirs(output_video_dir, exist_ok=True)

    input_filename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_filename = f"{input_filename}_analyzed_output.mp4"
    output_video_path = os.path.join(output_video_dir, output_video_filename)


    sv.process_video(
        source_path=input_video_path,
        target_path=output_video_path,
        callback=callback
    )
    avg_flocking_index = np.nanmean(flocking_index_values) if flocking_index_values else 0
    avg_speed = np.nanmean(average_speeds) if average_speeds else 0
    avg_triangle_area = np.nanmean(average_triangle_areas) if average_triangle_areas else 0
    avg_edge_length = np.nanmean(average_triangle_edge_lengths) if average_triangle_edge_lengths else 0

    return {
        "flocking_index": f"{avg_flocking_index:.2f}",
        "avg_group_speed": f"{avg_speed:.2f}",
        "avg_ta": f"{avg_triangle_area/100:.2f} px²",
        "avg_tel": f"{avg_edge_length/1000:.2f} px",
        "output_video_path": output_video_path
    }

    # # Print results
    # print(f"Flocking Index (average): {avg_flocking_index}")
    # print(f"Average Speed (average): {avg_speed}")
    # print(f"Average Triangle Area (average): {avg_triangle_area/100:.2f} px²")
    # print(f"Average Triangle Edge Length (average): {avg_edge_length/1000:.2f} px")
    # print(f"Output Video Path: {output_video_path}")


# process_video("tank_a_fed.MP4")
