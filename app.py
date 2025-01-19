from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from ultralytics import YOLO
from datetime import datetime
import cv2
import os

app = Flask(__name__)

# Load pretrained YOLO model
model = YOLO("classifier.pt")

# filepath = "Shih-Tzu.jpeg"

@app.route("/classify-breed", methods=["POST"])
def predict_img():
  if request.method == "POST":
    if 'file' in request.files:
      file = request.files['file'] or request.files['captured_image']

      timestamp = datetime.now()
      formatted = int(timestamp.timestamp())
      filename = f"temp/{formatted}_{file.filename}"
      file_extension = file.filename.rsplit('.', 1)[1].lower()
      
      destination_path = f'temp/{formatted}_{file.filename}'
      file.save(destination_path)
      filepath = destination_path

      results = model(filepath, verbose=False,conf=0.25, show_boxes=True, box=True, line_width=2, imgsz=640, classes=[15, 16])
      result = results[0]

      orig_image = cv2.imread(filepath)
      font = cv2.FONT_HERSHEY_SIMPLEX
      img = cv2.resize(orig_image, (800, 640))

      probs = result.probs.top5[:1][::-1]

      breed_type = ""
      for i in range(len(probs)):
        class_id = probs[i]
        name = result.names[class_id]
        y = i + 10
        label_text = f"Breed type: {name}"
        height, width, channels = img.shape
        new_y = (height - y)
        position = (20, new_y - 27 * i)
        font_scale = 0.75
        font_color = (0, 0, 0)
        thickness = 2
        breed_type = name

        cv2.putText(img, label_text, position, font, font_scale, font_color, thickness)

      # Save the resulting image
      cv2.imwrite(filepath, img)

      return jsonify({
        "message" : breed_type,
        "filename" : filename
      })
    else:
      return jsonify({
        "message" : "File is not allowed by its mime."
      })
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT"))