from flask import Flask, render_template, request, redirect, url_for, Response, send_from_directory
import os
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

video_cap = None
model = None
output_video_path = ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_cap, model, processing_stopped

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
            output_path,text = process_image(file_path)
        else:
            output_path = process_video(file_path)
            output_path = output_path.replace('static/', '')
            return redirect(url_for('video_feed', file_path=output_path))
        output_path = output_path.replace('static/', '')
        return render_template('index.html', file_path=output_path, text= text)
    return redirect(request.url)

@app.route('/download/<path:filename>')
def download_file(filename):
    print(filename)
    return send_from_directory('static/', filename, as_attachment=True)

def process_image(image_path):
    image = cv2.imread(image_path)
    
    model = YOLO(r'C:\Users\aryan\OneDrive\Desktop\CV PROJECTS\number plate recognition\best.pt')

    results = model(image)
    
    annotator = Annotator(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0] 
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    image = annotator.result()
    x,y,w,h = b
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    img_crop = image[y:h, x:w]
    
    output_path = image_path.replace('uploads', 'uploads/processed')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img_crop)
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    import easyocr
    reader = easyocr.Reader(['en'])
    result = reader.readtext(gray)
    text = ""

    for detection in result:
        if(len(results)==1) or (len(detection[1]>6) and detection[2]>0.2):
            text = detection[1]
            text = (str(text.replace('*','').replace('.','').replace(' ','').upper()))
    return output_path,text
    
def process_video(video_path):
    global video_cap
    video_cap = cv2.VideoCapture(video_path)
    if not video_cap.isOpened():
        print("Error opening video stream or file")
        return None

    return video_path.replace('static/', '')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global video_cap, model

    model = YOLO(r'C:\Users\aryan\OneDrive\Desktop\CV PROJECTS\number plate recognition\best.pt')

    org = (450, 250)  # Coordinates of the bottom-left corner of the text string
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    color = (0, 255, 0)  # BGR color (blue, green, red)
    thickness = 4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join('static/uploads/processed', 'output.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    crop_img =None
    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  
                c = box.cls
                x,y,w,h = b
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                annotator.box_label(b, model.names[int(c)])
                crop_img = frame[y:y+h, x:x+w]
            annotated_frame = annotator.result()
            
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            import easyocr

            reader = easyocr.Reader(['en'])

            result = reader.readtext(gray)
            text = ""

            for detection in result:
                if(len(results)==1) or (len(detection[1]>6) and detection[2]>0.2):
                    text = detection[1]
                    text = (str(text.replace('*','').replace('.','').replace(' ','').upper()))
                
        # Get the annotated frame 
        
        cv2.putText(annotated_frame, text, org, font, font_scale, color, thickness)
        out.write(annotated_frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    video_cap.release()
    out.release()

if __name__ == '__main__':
    app.run(debug=True)
