import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import random
import time

from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.cluster import DBSCAN

debug_mode = False

############################################################################################################
class Perception:
    def __init__(self, model_name='faster_rcnn_inception_v2_coco_2017_11_08', num_classes=90):
        global debug_mode
        self.model_name = model_name
        self.num_classes = num_classes
        self.path_to_model = os.path.join(self.model_name, 'saved_model')
        self.path_to_labels = os.path.join(self.model_name, 'mscoco_label_map.pbtxt')

        self.model = self._load_model()
        self.category_index = self._load_label_map()

        self.tracker = DeepSort(max_age=60, n_init=1)

        self.previous_ids = {}
        self.previous_bboxes = self.current_bboxes.copy() if hasattr(self, "current_bboxes") else {}

        self.current_ids = {}

        self.failure_counts = {} 
        self.stop_tracking_ids = set() 
        self.FAILURE_THRESHOLD = 3

    def _load_model(self):
        print(f"🔄 Loading model from {self.path_to_model} ...")
        model = tf.saved_model.load(self.path_to_model)
        print("✅ Model loaded successfully!")
        return model

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True
        )
        return label_map_util.create_category_index(categories)

    def run_inference(self, image, threshold):
        image_tensor = tf.convert_to_tensor(np.expand_dims(image, axis=0), dtype=tf.uint8)

        model_fn = self.model.signatures["serving_default"]
        output_dict = model_fn(image_tensor)

        num_detections = int(output_dict["num_detections"].numpy()[0])
        detection_boxes = output_dict["detection_boxes"].numpy()[0][:num_detections]
        detection_scores = output_dict["detection_scores"].numpy()[0][:num_detections]
        detection_classes = output_dict["detection_classes"].numpy()[0][:num_detections].astype(np.int64)

        valid_indices = np.where((detection_scores >= threshold)

        output_dict = {
            "num_detections": len(valid_indices[0]),
            "detection_boxes": detection_boxes[valid_indices],
            "detection_scores": detection_scores[valid_indices],
            "detection_classes": detection_classes[valid_indices],
        }
        
        return output_dict

    def visualize(self, image, output_dict, threshold):
        visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=threshold
        )
        return image

    def apply_nms(self, boxes, scores, classes, max_output_size=100, iou_threshold=0.4, score_threshold=0.3):
        if boxes.shape[0] == 0:
            return boxes, scores, classes

        selected_indices = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        nms_boxes = tf.gather(boxes, selected_indices)
        nms_scores = tf.gather(scores, selected_indices)
        nms_classes = tf.gather(classes, selected_indices)

        return nms_boxes.numpy(), nms_scores.numpy(), nms_classes.numpy()

    def detect(self, image, threshold, stats=None):
        h, w, _ = image.shape

        output_dict = self.run_inference(image, threshold)
        
        nms_boxes, nms_scores, nms_classes = self.apply_nms(output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes'])
        
        output_dict["detection_boxes"] = nms_boxes
        output_dict["detection_scores"] = nms_scores
        output_dict["detection_classes"] = nms_classes
        output_dict["num_detections"] = len(nms_boxes)

        self.visualize(image, output_dict, threshold)
        
        if stats is None: 
            if debug_mode:print("==========================================================================================")
        if output_dict['num_detections'] > 0:
            for i in range(output_dict['num_detections']):
                score = output_dict['detection_scores'][i]
                class_id = output_dict['detection_classes'][i]
                y1, x1, y2, x2 = output_dict['detection_boxes'][i]
                x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)
                class_name = self.category_index[class_id]['name']

            if stats is None: 
                if debug_mode:print(f"Detection Result:  - Class: {class_name}, Box: {x1, y1, x2, y2}, Score: {score:.2f}")

                if stats is not None:
                    stats["total_num"][class_name] = stats["total_num"].get(class_name, 0) + 1

                    if score > stats["max_score"]:
                        stats["max_score"] = score
                        stats["max_class_score"] = class_name
        else:
            if stats is None: 
                if debug_mode:print("No detections found.")

        return (image, stats) if stats is not None else (image, output_dict)
        
    def tracking(self, image, output_dict): 
        self.previous_ids = self.current_ids.copy()
        self.current_ids = {}

        h, w, _ = image.shape
        detections = []
        stop_tracking_ids = getattr(self, "stop_tracking_ids", set())

        for i in range(output_dict['num_detections']):
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
            bbox = (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)
            score = output_dict['detection_scores'][i]
            class_id = output_dict['detection_classes'][i]
            detections.append(([bbox[0], bbox[1], bbox[2], bbox[3]], score, class_id))

        tracks = self.tracker.update_tracks(detections, frame=image)

        time_update_map = {}
        for track in tracks:
            if track.is_confirmed():
                time_update_map[track.track_id] = track.time_since_update

        boxes, centers, ids, classes = [], [], [], []

        for track in tracks:
            if not track.is_confirmed():
                continue
            l, t, r, b = track.to_ltrb()
            cx, cy = (l + r) / 2, (t + b) / 2
            boxes.append([l, t, r, b])
            centers.append([cx, cy])
            ids.append(track.track_id)
            classes.append(track.det_class if hasattr(track, "det_class") else "Unknown")

        merged_ids = set()
        for cls in set(classes):
            cls_indices = [i for i, c in enumerate(classes) if c == cls]
            if len(cls_indices) <= 1:
                i = cls_indices[0]
                self._save_track_result(image, ids[i], boxes[i], classes[i], h, w, time_update_map.get(ids[i], -1))
                continue

            cls_centers = np.array([centers[i] for i in cls_indices])
            cls_boxes = [boxes[i] for i in cls_indices]
            cls_ids = [ids[i] for i in cls_indices]
            
            start = time.perf_counter()
            clustering = DBSCAN(eps=200, min_samples=1).fit(cls_centers)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            print(f"clustering time: {elapsed:.6f} ms")
            for cluster_id in set(clustering.labels_):
                cluster_indices = [i for i, label in enumerate(clustering.labels_) if label == cluster_id]
                cluster_boxes = [cls_boxes[i] for i in cluster_indices]
                cluster_ids = [cls_ids[i] for i in cluster_indices]

                l = min(b[0] for b in cluster_boxes)
                t = min(b[1] for b in cluster_boxes)
                r = max(b[2] for b in cluster_boxes)
                b = max(b[3] for b in cluster_boxes)

                rep_id = min(cluster_ids)
                merged_ids.update(cluster_ids)

                self._save_track_result(image, rep_id, [l, t, r, b], cls, h, w, time_update_map.get(rep_id, -1))

        for i in range(len(ids)):
            if ids[i] not in merged_ids:
                self._save_track_result(image, ids[i], boxes[i], classes[i], h, w, time_update_map.get(ids[i], -1))

        return image

    def _save_track_result(self, image, track_id, box, cls, h, w, time_since_update):
        l, t, r, b = box
        margin_factor = 0.5
        margin_x = (r - l) * margin_factor
        margin_y = (b - t) * margin_factor

        pred_bbox = [
            max(0, l - margin_x),
            max(0, t - margin_y),
            min(w, r + margin_x),
            min(h, b + margin_y)
        ]

        self.previous_bboxes[track_id] = (
            round(pred_bbox[0]), round(pred_bbox[1]),
            round(pred_bbox[2]), round(pred_bbox[3])
        )

        cv2.rectangle(image, (int(pred_bbox[0]), int(pred_bbox[1])),
                            (int(pred_bbox[2]), int(pred_bbox[3])), (0,255,0), 2)
        cv2.putText(image, f'ID {track_id}', (int(pred_bbox[0]), int(pred_bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        
        if time_since_update == 0:
            if debug_mode:print(f"🔵 Detection: ID={track_id}, Class={cls}, Box={box}")
            self.current_ids[track_id] = cls
        else:
            if debug_mode:print(f"🟡 max_age: ID={track_id}, Class={cls}, Box={box}, time_since_update={time_since_update}")

    def consistency_check(self, image):
        start = time.perf_counter()
        util = Utility()
        ids = {track_id: self.previous_ids[track_id] for track_id in set(self.previous_ids.keys()) - set(self.current_ids.keys())}
        
        if ids:
            if debug_mode: print(f"⚠️ Warning: The following IDs disappeared unexpectedly: {sorted(ids.keys())}")
            util.display_warning(image)
        
        for track_id, new_class in self.current_ids.items():
            if track_id in self.previous_ids and self.previous_ids[track_id] != new_class:
                if debug_mode: print(f"⚠️ Warning: Object ID {track_id} changed from {self.previous_ids[track_id]} to {new_class}")
                ids[track_id] = (self.previous_ids[track_id])
                util.display_warning(image)
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        print(f"consistency_check time: {elapsed:.6f} ms")
        return image, ids
    
############################################################################################################

class Utility:
    def plot(self, input_dir, output_dir, detection_data, filename, total_frames):
        fig, ax = plt.subplots(figsize=(12, 1.5))
        
        ax.set_xlim(1, total_frames)
        ax.set_ylim(0, 1)

        ax.set_xticks(range(1, total_frames + 1))
        ax.set_xticklabels([str(i) for i in range(1, total_frames + 1)])

        for frame_idx, detected in enumerate(detection_data.values(), start=1):
            color = "#FF9999" if "stop sign" in detected else "#99CCFF"
            ax.add_patch(plt.Rectangle((frame_idx, 0), 1, 1, color=color))

        first_detect_frame = next(
            (frame_idx for frame_idx, detected in enumerate(detection_data.values(), start=1) if "stop sign" in detected),
            None
        )

        if first_detect_frame:
            ax.axvline(x=first_detect_frame, color='blue', linestyle='--', linewidth=1)  
            ax.text(first_detect_frame, 0.5, "Detect", color='blue', fontsize=10, fontweight='bold',
                    verticalalignment='center', horizontalalignment='center')

        ax.set_yticks([])
        ax.set_frame_on(False)

        output_image_path = f"{output_dir}/stop_sign_detection_{filename}.png"
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)


    def display_warning(self, image):
        warning_text = " WARNING: Anomaly Detected!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 0, 0)
        bg_color = (255, 255, 255)
        thickness = 2
        x, y = 20, 50

        (text_width, text_height), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
        cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), bg_color, -1)
        cv2.putText(image, warning_text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    def display_success(self, image):
        success_text = "Correction Success!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 128, 0) 
        bg_color = (255, 255, 255)  
        thickness = 2
        h, w, _ = image.shape
        margin = 10

        (text_width, text_height), _ = cv2.getTextSize(success_text, font, font_scale, thickness)
        
        x = w - text_width - margin
        y = h - margin

        cv2.rectangle(image, 
                    (x - 5, y - text_height - 5), 
                    (x + text_width + 5, y + 5), 
                    bg_color, -1)

        cv2.putText(image, success_text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    def display_failure(self, image):
        failure_text = "Correction Failed!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 0, 255)  
        bg_color = (255, 255, 255)  
        thickness = 2

        h, w, _ = image.shape
        margin = 10

        (text_width, text_height), _ = cv2.getTextSize(failure_text, font, font_scale, thickness)

        x = w - text_width - margin
        y = h - margin

        cv2.rectangle(image, 
                    (x - 5, y - text_height - 5), 
                    (x + text_width + 5, y + 5), 
                    bg_color, -1)

        cv2.putText(image, failure_text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    def draw_plot(self, input_dir, output_dir, stop_sign_detection, filename):
        self.plot(input_dir, output_dir, stop_sign_detection, filename, 80)
        name = f"stop_sign_detection_{filename}.json"
        json_file = os.path.join(output_dir, name)
        with open(json_file, "w") as f:
            json.dump(stop_sign_detection, f, indent=4)

    def get_class_id_from_name(self, class_name, category_index):
        for class_id, class_info in category_index.items():
            if class_info["name"] == class_name:
                return class_id
    
############################################################################################################

class Transformation:
    def resize(self, image, resize_factor):
        h, w, _ = image.shape
        new_h, new_w = int(h * resize_factor), int(w * resize_factor)

        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


    def rotate(self, image, angle, axis):
        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2

        if axis.lower() == "z":
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        angle_rad = np.deg2rad(angle)

        if axis.lower() == "x":
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ], dtype=np.float32)
        elif axis.lower() == "y":
            rotation_matrix = np.array([
                [np.cos(angle_rad), 0, np.sin(angle_rad)],
                [0, 1, 0],
                [-np.sin(angle_rad), 0, np.cos(angle_rad)]
            ], dtype=np.float32)
        else:
            raise ValueError("axis should be 'x', 'y', or 'z'.")

        points_3d = np.array([
            [-center_x, -center_y, 0],  
            [center_x, -center_y, 0],   
            [center_x, center_y, 0],    
            [-center_x, center_y, 0]
        ], dtype=np.float32)

        rotated_points = np.dot(points_3d, rotation_matrix.T)
        projected_points = rotated_points[:, :2] + np.array([center_x, center_y])

        src_points, dst_points = points_3d[:, :2] + np.array([center_x, center_y]), projected_points
        warp_matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))

        return cv2.warpPerspective(image, warp_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


    def adjust_brightness(self, image, beta):
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

    def transformation(self, image, bounding_box, resize_factor, x_angle, y_angle, z_angle, beta, save_path="transformation_steps"):
        os.makedirs(save_path, exist_ok=True)

        h, w, _ = image.shape
        ymin, xmin, ymax, xmax = bounding_box
        left, right = max(int(xmin * w), 0), min(int(xmax * w), w)
        top, bottom = max(int(ymin * h), 0), min(int(ymax * h), h)

        cropped_object = image[top:bottom, left:right]

        resized_object = self.resize(cropped_object, resize_factor)
        cv2.imwrite(os.path.join(save_path, "1_resized.jpg"), resized_object)

        adjusted_object = self.adjust_brightness(resized_object, beta)
        cv2.imwrite(os.path.join(save_path, "2_brightness_adjusted.jpg"), adjusted_object)

        x_rotated_object = self.rotate(adjusted_object, x_angle, 'x')
        cv2.imwrite(os.path.join(save_path, "3_x_rotated.jpg"), x_rotated_object)

        y_rotated_object = self.rotate(x_rotated_object, y_angle, 'y')
        cv2.imwrite(os.path.join(save_path, "4_y_rotated.jpg"), y_rotated_object)

        z_rotated_object = self.rotate(y_rotated_object, z_angle, 'z')
        cv2.imwrite(os.path.join(save_path, "5_z_rotated.jpg"), z_rotated_object)

        final_object = z_rotated_object
        cv2.imwrite(os.path.join(save_path, "6_final.jpg"), final_object)

        return final_object

    def setup_canvas(self, transform_object, image):
        h, w, _ = image.shape
        canvas = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

        canvas_center_x, canvas_center_y = w // 2, h // 2

        obj_h, obj_w = transform_object.shape[:2]
        obj_center_x, obj_center_y = obj_w // 2, obj_h // 2

        start_x = canvas_center_x - obj_center_x
        start_y = canvas_center_y - obj_center_y
        end_x = start_x + obj_w
        end_y = start_y + obj_h

        crop_x1 = max(0, -start_x) 
        crop_y1 = max(0, -start_y)  
        crop_x2 = obj_w - max(0, end_x - w) 
        crop_y2 = obj_h - max(0, end_y - h)  

        cropped_object = transform_object[crop_y1:crop_y2, crop_x1:crop_x2]

        paste_x1 = max(0, start_x)
        paste_y1 = max(0, start_y)
        paste_x2 = paste_x1 + cropped_object.shape[1]
        paste_y2 = paste_y1 + cropped_object.shape[0]

        canvas[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_object

        return canvas

############################################################################################################ 

class Defense:
    def __init__(self):
        self.transformation = Transformation()

        self.repeat_count =  10

        self.small_size_params = {
            "min": 20, "max": 40, "interval_size": 5
        }
        self.large_size_params = {
            "min": 40, "max": 60, "interval_size": 5
        }

        self.size_params = {
            "min": 30, "max": 50, "interval_size": 5
        }

        self.angle_params = {
            "min_x": -40, "max_x": 40, "interval_x": 2,
            "min_y": -40, "max_y": 40, "interval_y": 2,
            "min_z": -20, "max_z": 20, "interval_z": 2
        }

        self.beta_params = {
            # "min": -120, "max": -90, "interval": 5
            "min": -90, "max": -30, "interval": 5
        }

        self.stats = { 
            "total_num": {}, "max_score": 0, "max_class_score": None 
        }

    def load_image(self, image, anomaly_boxes, margin_ratio=0.2):
        height, width, _ = image.shape

        xmin, ymin, xmax, ymax = anomaly_boxes

        box_width = xmax - xmin
        box_height = ymax - ymin
        box_area = box_width * box_height

        margin_x = box_width * margin_ratio
        margin_y = box_height * margin_ratio

        xmin_expanded = int(max(0, xmin - margin_x))
        ymin_expanded = int(max(0, ymin - margin_y))
        xmax_expanded = int(min(width, xmax + margin_x))
        ymax_expanded = int(min(height, ymax + margin_y))\

        normalized_box = [
            ymin_expanded / height,
            xmin_expanded / width,
            ymax_expanded / height,
            xmax_expanded / width,
        ]

        roi = image[ymin_expanded:ymax_expanded, xmin_expanded:xmax_expanded]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = int(np.mean(gray_roi))

        if debug_mode:print(f"Anomaly Box - Area: {box_area}px², Brightness: {brightness}beta")

        return image, normalized_box, box_area, brightness


    def cal(self):
        total_num = self.stats["total_num"]
        max_class_score = self.stats["max_class_score"]
        max_score = self.stats["max_score"]

        if debug_mode: print("\n########################################################################################################################\n")
        if debug_mode: print("Total Detected Class:")
        total_detections = sum(self.stats["total_num"].values())

        max_class_num = None
        max_ratio = 0

        for class_name, count in total_num.items():
            class_ratio = (count / total_detections) * 100 if total_detections > 0 else 0
            if debug_mode: print(f"  - {class_name}: {count} ({class_ratio:.2f}%)")

            if class_ratio > max_ratio:
                max_ratio = class_ratio
                max_class_num = class_name

        if max_class_num:
            if debug_mode:print(f"\nClass with the Highest Ratio: {max_class_num} ({max_ratio:.2f}%)")

        if max_class_score:
            if debug_mode:print(f"Class with the Highest Score: {max_class_score} ({max_score:.2f})")

        if debug_mode:print(f"[{max_class_num}]")
        return max_class_num, total_detections


    def save_image(self, file_name, output_dir, re_detection_image, resize_factor=None, x_rotate_angle=None, y_rotate_angle=None, z_rotate_angle=None, beta_value=None):
        folder_path = os.path.join(output_dir, file_name)
        os.makedirs(folder_path, exist_ok=True)
        
        transformation_parts = []
        if resize_factor is not None:
            transformation_parts.append(f"size({resize_factor:.1f})")
        if x_rotate_angle is not None:
            transformation_parts.append(f"x({x_rotate_angle})")
        if y_rotate_angle is not None:
            transformation_parts.append(f"y({y_rotate_angle})")
        if z_rotate_angle is not None:
            transformation_parts.append(f"z({z_rotate_angle})")
        if beta_value is not None:
            transformation_parts.append(f"beta({beta_value})")
        
        transformation_file_name = "_".join(transformation_parts) + ".jpg"
        re_detection_path = os.path.join(folder_path, transformation_file_name)
        
        cv2.imwrite(re_detection_path, re_detection_image)
        if debug_mode: print(f"✅ Saved Transformed Image: {re_detection_path}")
    
    def run(self, image, anomaly_boxes, file_name, perception):
        start_time = time.time()
        
        image, normalized_box, box_area, brightness = self.load_image(image, anomaly_boxes)

        for i in range(self.repeat_count):
            start_time_t = time.time()
            start_time_i = time.time()

            size = random.choice(range(self.size_params["min"], self.size_params["max"], self.size_params["interval_size"]))
            x_angle = random.choice(range(self.angle_params["min_x"], self.angle_params["max_x"], self.angle_params["interval_x"]))
            y_angle = random.choice(range(self.angle_params["min_y"], self.angle_params["max_y"], self.angle_params["interval_y"]))
            z_angle = random.choice(range(self.angle_params["min_z"], self.angle_params["max_z"], self.angle_params["interval_z"]))
            beta = random.choice(range(self.beta_params["min"], self.beta_params["max"], self.beta_params["interval"]))

            transformation_object = self.transformation.transformation(image, normalized_box, size/10, 0, y_angle, 0, beta)
            transformation_frame = self.transformation.setup_canvas(transformation_object, image)

            end_time_transformation = time.time()

            re_detection_image, output_dict = perception.detect(transformation_frame, threshold=0.3, stats=self.stats)
            end_time_inferecne = time.time()
            elapsed_t = end_time_transformation - start_time_t
            elapsed_i = end_time_inferecne - start_time_i

            print(f"Transformatoin took {elapsed_t*1000:.1f} ms")
            print(f"Inference took {elapsed_i*1000:.1f} ms")

            self.save_image(file_name, "C:/Users/detect/transformed_frame", re_detection_image, resize_factor=size/10, x_rotate_angle=0, y_rotate_angle=y_angle, z_rotate_angle=0, beta_value=beta)
        calibrated_class, total_detections = self.cal()

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Defense module took {elapsed*1000:.1f} ms for file: {file_name}")

        return calibrated_class, total_detections
    
############################################################################################################ 

def main(input_dir, output_dir, option):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"❌ Error: Input folder '{input_dir}' not found.")
        return

    util = Utility()
    perception = Perception()
    defense = Defense()

    detection_before = {}
    detection_after = {}
    total_num = 0
    count = 1
    run_count = 0
    
    for file_name in sorted(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image = cv2.imread(file_path)  # BGR (save)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB (detect)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB (detect)

        if image is None:
            print(f"❌ Error: Unable to read image '{file_name}'")
            continue

        # Object Detection & Tracking
        output_image, output_dict = perception.detect(image, threshold=0.3)
        output_image = perception.tracking(output_image, output_dict)

        # Consistency Check
        output_image, ids = perception.consistency_check(output_image)

        # Before Defense
        detection_before2 = [
            perception.category_index[class_id]['name']
            for class_id in output_dict["detection_classes"]
        ]
        detection_before[count] = detection_before2

        # Defense Execution
        for id in ids:
            if id in perception.current_ids and perception.current_ids[id] == 13:
                continue
            
            defense = Defense()
            anomaly_boxes = perception.previous_bboxes[id]
            calibrated_class, total_detections = defense.run(image2, anomaly_boxes, file_name, perception)
            total_num += total_detections
            run_count += 1
            if debug_mode:print(f"Total: {total_num}")
            
            if calibrated_class == "stop sign":
                calibrated_class_id = util.get_class_id_from_name(calibrated_class, perception.category_index)
                output_dict["detection_classes"] = [calibrated_class_id]
                perception.current_ids[id] = calibrated_class_id
                util.display_success(image)
            elif calibrated_class is not None:
                calibrated_class_id = util.get_class_id_from_name(calibrated_class, perception.category_index)
                output_dict["detection_classes"] = [calibrated_class_id]
                perception.current_ids[id] = calibrated_class_id
                util.display_failure(image)
            else:
                pass
                
        # After Defense
        detection_after2 = [
            perception.category_index[class_id]['name']
            for class_id in output_dict["detection_classes"]
        ]
        detection_after[count] = detection_after2
        count += 1

        if debug_mode:print(f"\nPrevious Frame IDs: {perception.previous_ids}")
        if debug_mode:print(f"Current Frame IDs: {perception.current_ids}")

        output_file_path = os.path.join(output_dir, file_name)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # BGR (save)
        cv2.imwrite(output_file_path, output_image)
        if debug_mode:print(f"✅ Saved Detection Result: {output_file_path}")

    util.draw_plot(input_dir, output_dir, detection_before, "before")
    util.draw_plot(input_dir, output_dir, detection_after, "after")
    if debug_mode:print(f"Run Count: {run_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("input_dir", type=str, help="Path to the input directory", nargs="?", default="/home/cpss/")
    parser.add_argument("output_dir", type=str, help="Path to the output directory", nargs="?", default="/home/cpss/")
    parser.add_argument("option", type=str, help="Transformation Option", nargs="?", default="all")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.option)
