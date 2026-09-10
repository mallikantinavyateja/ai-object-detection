from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

model = YOLO("yolo11n.pt")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    image = request.files["image"]

    input_path = os.path.join("uploads", image.filename)
    image.save(input_path)

    results = model(input_path)

    result_image = results[0].plot()

    output_path = os.path.join(
        "static/results",
        image.filename
    )

    cv2.imwrite(output_path, result_image)

    # Get detected objects
    detected_objects = []

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0]) * 100

        object_name = model.names[class_id]

        detected_objects.append({
            "name": object_name,
            "confidence": round(confidence, 2)
        })

    object_count = len(detected_objects)

    return render_template(
        "index.html",
        result_image="results/" + image.filename,
        detected_objects=detected_objects,
        object_count=object_count
    )


if __name__ == "__main__":
    app.run(debug=True)