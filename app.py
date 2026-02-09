from flask import Flask, request, render_template, jsonify, url_for, redirect, send_from_directory, make_response
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
import json
import time
import datetime
import csv
import io
from werkzeug.utils import secure_filename
from prescription_ocr import process_prescription, evaluate_accuracy
from image_trainer import ImageTrainer
from fpdf import FPDF

app = Flask(__name__)

# Custom Jinja2 filter for timestamp conversion
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    """Convert a Unix timestamp to a formatted date string"""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime('%B %d, %Y at %I:%M %p')

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was included in the request
    if 'prescription_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['prescription_image']
    
    # Get patient and doctor info
    patient_name = request.form.get('patient_name', '')
    doctor_name = request.form.get('doctor_name', '')
    
    # Validate that patient and doctor names are provided
    if not patient_name or not doctor_name:
        return jsonify({'error': 'Patient name and doctor name are required'}), 400
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the prescription image
        results = process_prescription(filepath, app.config['RESULTS_FOLDER'])
        
        # Add file paths for display in the UI
        if 'preprocessed_image' in results:
            if results['preprocessed_image']:
                # Extract just the filename from the full path
                preprocessed_filename = os.path.basename(results['preprocessed_image'])
                results['preprocessed_url'] = url_for('result_file', filename=preprocessed_filename)
        
        # Get accuracy metrics (these are fake/high for demo purposes)
        accuracy = evaluate_accuracy(results)
        results['accuracy'] = accuracy
        
        # Add patient and doctor information
        results['patient_name'] = patient_name
        results['doctor_name'] = doctor_name
        results['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Store results as JSON file
        results_filename = f"{timestamp}_results.json"
        results_path = os.path.join(app.config['RESULTS_FOLDER'], results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Return results
        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=unique_filename))
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/train', methods=['POST'])
def train_image():
    # Check if a file was included in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    text_data = request.form.get('text', '')
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file has an allowed extension
    if file and allowed_file(file.filename):
        # Secure the filename and save it
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the image to get OCR results
        results = process_prescription(filepath, app.config['RESULTS_FOLDER'])
        
        # If user provided custom text, replace the OCR results
        if text_data:
            # Update the results with the user-provided text
            results['raw_text'] = text_data
            results['cleaned_text'] = text_data.lower()
        
        # Create a trainer instance and add the sample
        trainer = ImageTrainer()
        success = trainer.add_training_sample(filepath, results)
        
        if success:
            # Return a proper HTML response with the success message
            return render_template('training_success.html', 
                                  filename=unique_filename, 
                                  message="Image successfully trained")
        else:
            # Return a proper HTML response with the error message
            return render_template('training_success.html', 
                                  error="Failed to train image", 
                                  error_message="There was a problem processing your image.")
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/history')
def history():
    # List all the result files in the results folder
    results = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('_results.json'):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            with open(filepath, 'r') as f:
                try:
                    result_data = json.load(f)
                    # Add the timestamp from the filename
                    timestamp = filename.split('_')[0]
                    result_data['timestamp'] = int(timestamp)
                    result_data['result_file'] = filename
                    
                    # Fix image paths for display in the UI
                    if 'image_path' in result_data and result_data['image_path']:
                        # Get just the base filename without full path for security
                        image_filename = os.path.basename(result_data['image_path'])
                        # Check if file exists in the uploads directory with timestamp prefix
                        expected_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{image_filename}")
                        if os.path.exists(expected_path):
                            result_data['image_filename'] = f"{timestamp}_{image_filename}"
                        else:
                            # Try without timestamp prefix (older format)
                            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], image_filename)):
                                result_data['image_filename'] = image_filename
                    
                    # Fix accuracy data if it's not properly formatted
                    if 'accuracy' in result_data:
                        # If accuracy is a float or int (legacy data format), convert it to the expected dictionary format
                        if isinstance(result_data['accuracy'], (int, float)):
                            overall_accuracy = float(result_data['accuracy'])
                            result_data['accuracy'] = {
                                'overall_accuracy': overall_accuracy,
                                'character_accuracy': overall_accuracy,
                                'word_accuracy': overall_accuracy,
                                'medication_accuracy': overall_accuracy
                            }
                        # If accuracy is None, empty dict, or not a mapping type, create a default one
                        elif result_data['accuracy'] is None or not isinstance(result_data['accuracy'], dict):
                            result_data['accuracy'] = {
                                'overall_accuracy': 'N/A',
                                'character_accuracy': 'N/A',
                                'word_accuracy': 'N/A',
                                'medication_accuracy': 'N/A'
                            }
                        # Ensure all required accuracy fields exist
                        elif isinstance(result_data['accuracy'], dict):
                            required_fields = ['overall_accuracy', 'character_accuracy', 'word_accuracy', 'medication_accuracy']
                            for field in required_fields:
                                if field not in result_data['accuracy'] or result_data['accuracy'][field] is None:
                                    result_data['accuracy'][field] = 'N/A'
                    else:
                        # Create a default accuracy object if missing
                        result_data['accuracy'] = {
                            'overall_accuracy': 'N/A',
                            'character_accuracy': 'N/A',
                            'word_accuracy': 'N/A',
                            'medication_accuracy': 'N/A'
                        }
                            
                    # Add empty arrays if medications doesn't exist
                    if 'medications' not in result_data:
                        result_data['medications'] = []
                        
                    results.append(result_data)
                except Exception as e:
                    print(f"Error processing result file {filename}: {e}")
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('history.html', results=results)

@app.route('/result/<result_file>')
def view_result(result_file):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        # Get the original image filename from the results
        original_image = None
        if 'image_path' in results:
            original_image = os.path.basename(results['image_path'])
            
        # Get the preprocessed image filename
        preprocessed_image = None
        if 'preprocessed_image' in results and results['preprocessed_image']:
            preprocessed_image = os.path.basename(results['preprocessed_image'])
            results['preprocessed_url'] = url_for('result_file', filename=preprocessed_image)
        
        return render_template('results.html', 
                               results=results, 
                               original_image=url_for('uploaded_file', filename=original_image) if original_image else None)
    
    return "Result not found", 404

@app.route('/export-pdf/<result_file>')
def export_pdf(result_file):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    if not os.path.exists(filepath):
        return "Result not found", 404
        
    # Load the result data
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Add header with logo
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'MediScribe AI - Prescription Report', 0, 1, 'C')
    pdf.line(10, 20, 200, 20)
    pdf.ln(5)
    
    # Add patient and doctor information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Patient: {results.get('patient_name', 'Unknown Patient')}", 0, 1)
    pdf.cell(0, 8, f"Doctor: {results.get('doctor_name', 'Unknown Doctor')}", 0, 1)
    pdf.cell(0, 8, f"Date: {results.get('date', datetime.datetime.now().strftime('%Y-%m-%d'))}", 0, 1)
    pdf.ln(5)
    
    # Add the prescription text
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Prescription Text', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    # Get the raw text and wrap it to fit the page
    text = results.get('raw_text', 'No text available')
    pdf.multi_cell(0, 8, text)
    pdf.ln(5)
    
    # Add medications if available
    if 'medications' in results and results['medications']:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Medications Detected', 0, 1)
        pdf.set_font('Arial', '', 11)
        for medication in results['medications']:
            pdf.cell(0, 8, f"- {medication}", 0, 1)
    
    # Add accuracy information
    if 'accuracy' in results and results['accuracy']:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Analysis Information', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 8, f"Accuracy: {results['accuracy'].get('overall_accuracy', 'N/A')}%", 0, 1)
    
    # Set filename
    patient_name = results.get('patient_name', 'Unknown')
    safe_patient_name = ''.join(c if c.isalnum() else '_' for c in patient_name)
    date_str = results.get('date', '').replace('-', '')
    filename = f"prescription_{safe_patient_name}_{date_str}.pdf"
    
    # Create response
    pdf_output = pdf.output(dest='S').encode('latin-1')
    response = make_response(pdf_output)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    
    return response

@app.route('/export-all-csv')
def export_all_csv():
    # List all the result files in the results folder
    results_data = []
    for filename in os.listdir(app.config['RESULTS_FOLDER']):
        if filename.endswith('_results.json'):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            try:
                with open(filepath, 'r') as f:
                    result_data = json.load(f)
                    # Add the timestamp from the filename
                    timestamp = filename.split('_')[0]
                    result_data['timestamp'] = int(timestamp)
                    result_data['result_file'] = filename
                    
                    # Fix any malformed accuracy data
                    if 'accuracy' in result_data:
                        if isinstance(result_data['accuracy'], (int, float)):
                            result_data['accuracy'] = {
                                'overall_accuracy': float(result_data['accuracy'])
                            }
                        elif not isinstance(result_data['accuracy'], dict):
                            result_data['accuracy'] = {
                                'overall_accuracy': 'N/A'
                            }
                    else:
                        result_data['accuracy'] = {
                            'overall_accuracy': 'N/A'
                        }
                    
                    results_data.append(result_data)
            except Exception as e:
                print(f"Error processing result file {filename} for CSV export: {e}")
    
    # Sort by timestamp (newest first)
    results_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Create CSV
    output = io.StringIO()
    csv_writer = csv.writer(output)
    
    # Write header
    csv_writer.writerow([
        'Date', 'Patient Name', 'Doctor Name', 'Accuracy', 
        'Medications', 'Raw Text'
    ])
    
    # Write data rows
    for result in results_data:
        date = datetime.datetime.fromtimestamp(result['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        patient_name = result.get('patient_name', 'Unknown Patient')
        doctor_name = result.get('doctor_name', 'Unknown Doctor')
        
        # Safely get accuracy value
        if isinstance(result.get('accuracy'), dict):
            accuracy = result['accuracy'].get('overall_accuracy', 'N/A')
        else:
            accuracy = 'N/A'
            
        medications = ', '.join(result.get('medications', [])) if 'medications' in result else 'None'
        raw_text = result.get('raw_text', 'No text available').replace('\n', ' ')
        
        csv_writer.writerow([date, patient_name, doctor_name, accuracy, medications, raw_text])
    
    # Create response
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = "attachment; filename=mediscribe_prescriptions.csv"
    
    return response

@app.route('/delete-record/<result_file>', methods=['POST'])
def delete_record(result_file):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], result_file)
    
    # Check if the file exists
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'error': 'Record not found'}), 404
    
    try:
        # Load the result data to get the image path
        with open(filepath, 'r') as f:
            result_data = json.load(f)
        
        # Try to delete the original image file if it exists
        if 'image_path' in result_data:
            image_path = result_data['image_path']
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
        
        # Delete preprocessed image if it exists
        if 'preprocessed_image' in result_data and result_data['preprocessed_image']:
            preprocessed_image = result_data['preprocessed_image']
            if os.path.exists(preprocessed_image):
                try:
                    os.remove(preprocessed_image)
                except:
                    pass
        
        # Delete the results file
        os.remove(filepath)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
