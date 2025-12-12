from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
from fpdf import FPDF
from PIL import Image

# NEW: database imports
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Initialize app
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = 'static/uploads/'
GRAPH_FOLDER = 'static/graphs/'
REPORT_FOLDER = 'reports/'
MODEL_PATH = 'models/final_paddy_model.h5'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# --- NEW: Database config ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'paddy.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- NEW: Prediction model (table) ---
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(200), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.id} {self.image_filename} {self.predicted_class} {self.confidence:.3f}>"

with app.app_context():
    db.create_all()

# Load model
model = load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ['BacterialLeafBlight ', ' Brown Spot', 'Healthy', 'Leaf Blast', 'leaf_scald', 'Other']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)

            # Convert WEBP → JPG if needed
            if img_path.lower().endswith('.webp'):
                im = Image.open(img_path).convert("RGB")
                new_path = img_path.rsplit('.', 1)[0] + '.jpg'
                im.save(new_path, "JPEG")
                img_path = new_path

            # Predict
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            preds = model.predict(x)[0]

            # Generate confidence graph
            plt.figure(figsize=(8, 5))
            plt.bar(CLASS_NAMES, preds, color='skyblue')
            plt.ylim([0, 1])
            plt.ylabel('Confidence')
            plt.title('Prediction Confidence')
            graph_path = os.path.join(GRAPH_FOLDER, file.filename.split('.')[0] + '_graph.png')
            plt.tight_layout()
            plt.savefig(graph_path)
            plt.close()

            # Generate PDF report (with smaller image)
            pdf_filename = file.filename.split('.')[0] + '_report.pdf'
            pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
            pdf = FPDF()
            pdf.add_page()

            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Paddy Leaf Disease Prediction Report", ln=True, align='C')
            pdf.ln(10)

            # Uploaded Image Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Uploaded Image:", ln=True)
            pdf.image(img_path, x=45, w=120)  # smaller image width
            pdf.ln(70)

            # Prediction Scores
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Prediction Confidence Scores:", ln=True)
            pdf.set_font("Arial", '', 12)
            pdf.ln(3)
            for cls, prob in zip(CLASS_NAMES, preds):
                pdf.cell(0, 10, f"- {cls}: {prob*100:.2f}%", ln=True)
            pdf.ln(10)

            # Graph Section
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "Prediction Graph:", ln=True)
            pdf.image(graph_path, x=30, w=150)

            pdf.output(pdf_path)

            predicted_class = CLASS_NAMES[np.argmax(preds)]
            top_confidence = float(np.max(preds))  # top confidence for DB

            # --- NEW: Save prediction to DB ---
            try:
                new_entry = Prediction(
                    image_filename=os.path.basename(img_path),
                    predicted_class=predicted_class,
                    confidence=top_confidence
                )
                db.session.add(new_entry)
                db.session.commit()
            except Exception as e:
                # Do not interrupt the normal flow if DB save fails; log to console
                print("DB save error:", e)

            return render_template('index.html',
                                   img_path=img_path,
                                   graph_path=graph_path,
                                   predicted_class=predicted_class,
                                   pdf_filename=pdf_filename)

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(REPORT_FOLDER, filename)
    return send_file(path, as_attachment=True)


# --- NEW: simple history page to view stored predictions ---
@app.route('/history')
def history():
    records = Prediction.query.order_by(Prediction.created_at.asc()).all()
    return render_template('history.html', records=records)


if __name__ == '__main__':
    app.run(debug=True)

