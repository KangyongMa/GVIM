from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from simulate_ai import get_chemistry_lab
import os
import logging
import json
import base64

app = Flask(__name__, static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = app.logger

# Initialize ChemistryLab
chemistry_lab = None
literature_path = ""
web_url_path = ""

# In-memory storage for chat history
chat_sessions = {}

def process_search_results(search_results):
    if isinstance(search_results, str):
        try:
            search_results = json.loads(search_results)
        except json.JSONDecodeError:
            return [{"content": search_results, "url": "N/A", "title": "Search Result"}]

    if isinstance(search_results, list):
        return [{
            "content": result.get("content", ""),
            "url": result.get("url", "N/A"),
            "title": result.get("title", "Search Result")
        } for result in search_results]
    elif isinstance(search_results, dict):
        return [{
            "content": search_results.get("content", ""),
            "url": search_results.get("url", "N/A"),
            "title": search_results.get("title", "Search Result")
        }]
    else:
        return [{"content": str(search_results), "url": "N/A", "title": "Search Result"}]

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

        for message in response:
            if message['role'] == 'assistant' and '[WEB_SEARCH_SUMMARY:' in message['content']:
                search_start = message['content'].index('[WEB_SEARCH_SUMMARY:')
                search_end = message['content'].index(']', search_start)
                search_content = message['content'][search_start+18:search_end]

                processed_results = process_search_results(search_content)

                message['content'] = message['content'][:search_start] + message['content'][search_end+1:]
                message['search_results'] = processed_results

        # Store the chat history
        session_id = request.form.get('session_id', 'default')
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        image_data_base64 = None
        if image_file:
            image_data = image_file.read()
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')

        chat_sessions[session_id].append({'user_input': user_input, 'response': response, 'image_data': image_data_base64})

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

@app.route('/history', methods=['GET'])
def history():
    session_id = request.args.get('session_id', 'default')
    return jsonify(chat_sessions.get(session_id, []))

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


