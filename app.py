from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from ultralytics import YOLO
from supabase import create_client, Client
from datetime import datetime
import cv2
import os

app = Flask(__name__)

# Load pretrained YOLO model
model = YOLO("classifier.pt")


# Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
bucket = os.environ.get("S3_BUCKET")
Client = create_client(url, key)


@app.route("/")
def index():
  return jsonify({
    "message": "api working"
  })

@app.route("/classify-breed", methods=["POST"])
def predict_img():
  if request.method == "POST":
    if 'file' in request.files:
      file = request.files['file'] or request.files['captured_image']

      timestamp = datetime.now()
      formatted = int(timestamp.timestamp())
      filename = f"uploads/{formatted}_{file.filename.lower()}"
      # file_extension = file.filename.rsplit('.', 1)[1].lower()
      
      destination_path = f'uploads/{formatted}_{file.filename.lower()}'
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
      
      # Upload file
      upload_file(filepath)
      
      # Get public link
      link = get_link(filename)
        
      return jsonify({
        "message" : breed_type,
        "breed_type" : breed_type,
        "file_name" : file.filename.lower(),
        "image_link" : link
      })
    else:
      return jsonify({
        "message" : "File is not allowed by its mime."
      })

# Upload a file to s3
def upload_file (filepath):
  with open(filepath, 'rb') as file:
    Client.storage.from_(bucket).upload(
        file=file,
        path=filepath,
        file_options={"cache-control": "3600", "upsert": "false", "content-type" : "image/*"},
    )
    # Delete file after uploaded to bucket
    if os.path.exists(filepath):
      print("exist.")
      os.remove(filepath)
    else:
      print("Does not exist.")
   
# Get a file link to s3
def get_link(filename):
  link = Client.storage.from_("stray").get_public_url(filename)
  return link

# Server entry
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    app.run(host="0.0.0.0", port=port)