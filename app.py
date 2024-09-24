from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from simulate_ai import get_chemistry_lab
import os
import logging
import json

app = Flask(__name__, static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# Initialize ChemistryLab
chemistry_lab = None
literature_path = ""
web_url_path = ""

@app.route('/configure', methods=['POST'])
def configure():
    global chemistry_lab, literature_path, web_url_path
    data = request.json
    literature_path = data.get('literature_path', '')
    web_url_path = data.get('web_url_path', '')
    chemistry_lab = get_chemistry_lab(literature_path)
    logger.info(f"Configured with literature_path: {literature_path}, web_url_path: {web_url_path}")
    return jsonify({'status': 'Configuration updated', 'literature_path': literature_path})

@app.route('/simulate', methods=['POST'])
def simulate():
    global chemistry_lab, web_url_path, literature_path
    if not chemistry_lab:
        chemistry_lab = get_chemistry_lab(literature_path)
    
    user_input = request.form.get('message', '')
    image_file = request.files.get('image')
    new_literature_path = request.form.get('literature_path', '')
    web_url_path = request.form.get('web_url_path', '')
    
    logger.info(f"Received request - User input: {user_input}, Literature path: {new_literature_path}, Web URL path: {web_url_path}")
    
    # Update literature_path if a new one is provided
    if new_literature_path and new_literature_path != literature_path:
        logger.info(f"Updating literature path from {literature_path} to {new_literature_path}")
        literature_path = new_literature_path
        chemistry_lab = get_chemistry_lab(literature_path)  # Recreate chemistry_lab with new literature_path
    
    try:
        response = chemistry_lab.process_user_input(user_input, image_data=image_file.read() if image_file else None, literature_path=literature_path, web_url_path=web_url_path)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing user input: {str(e)}", exc_info=True)
        return jsonify([{
            'role': 'assistant',
            'name': 'System',
            'content': f"Error processing your input: {str(e)}"
        }]), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    global chemistry_lab
    if not chemistry_lab:
        return jsonify({'error': 'ChemistryLab not configured'}), 400

    user_feedback = request.json['feedback']
    logger.info(f"Received feedback: {user_feedback}")
    try:
        chemistry_lab.get_user_feedback(user_feedback)
        return jsonify({'status': 'Feedback received'})
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('static/images', filename)

if __name__ == '__main__':
    app.run(debug=True)

