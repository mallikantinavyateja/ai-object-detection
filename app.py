from flask import Flask, render_template, request
import os
import cv2
import onnxruntime as ort
import numpy as np

app = Flask(__name__)

# Load YOLO11 ONNX model
session = ort.InferenceSession(
    "yolo11n.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)


# COCO class names
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image uploaded", 400

    image = request.files["image"]

    if image.filename == "":
        return "No file selected", 400

    input_path = os.path.join("uploads", image.filename)
    image.save(input_path)

    frame = cv2.imread(input_path)

    if frame is None:
        return "Could not read image", 400

    original_height, original_width = frame.shape[:2]

    # Resize image for YOLO
    resized = cv2.resize(frame, (640, 640))

    # Convert BGR to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    resized = resized.astype(np.float32) / 255.0

    # Change HWC to CHW
    resized = np.transpose(resized, (2, 0, 1))

    # Add batch dimension
    input_image = np.expand_dims(resized, axis=0)

    # Run ONNX model
    outputs = session.run(None, {input_name: input_image})

    predictions = outputs[0]

    # YOLO output is usually:
    # (1, 84, 8400)
    # Convert it to:
    # (8400, 84)
    if len(predictions.shape) == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.transpose()

    boxes = []
    confidences = []
    class_ids = []

    confidence_threshold = 0.15
    nms_threshold = 0.45

    x_scale = original_width / 640
    y_scale = original_height / 640

    for prediction in predictions:

        # First 4 values are:
        # center_x, center_y, width, height
        x_center, y_center, box_width, box_height = prediction[:4]

        # Remaining values are class scores
        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < confidence_threshold:
            continue

        # Convert center coordinates to corner coordinates
        x = int((x_center - box_width / 2) * x_scale)
        y = int((y_center - box_height / 2) * y_scale)

        width = int(box_width * x_scale)
        height = int(box_height * y_scale)

        boxes.append([x, y, width, height])
        confidences.append(confidence)
        class_ids.append(class_id)

    # Remove duplicate boxes
    indices = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        confidence_threshold,
        nms_threshold
    )

    detected_objects = []

    if len(indices) > 0:

        for index in indices.flatten():

            x, y, width, height = boxes[index]

            class_id = class_ids[index]
            confidence = round(confidences[index] * 100, 2)

            if class_id < len(class_names):
                object_name = class_names[class_id]
            else:
                object_name = "Unknown"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            # Draw label
            label = f"{object_name} {confidence}%"

            cv2.putText(
                frame,
                label,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            detected_objects.append({
                "name": object_name,
                "confidence": confidence
            })

    # Save result image
    output_path = os.path.join(
        "static/results",
        image.filename
    )

    cv2.imwrite(output_path, frame)

    object_count = len(detected_objects)

    return render_template(
        "index.html",
        result_image="results/" + image.filename,
        detected_objects=detected_objects,
        object_count=object_count
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )