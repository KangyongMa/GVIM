from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdMolDescriptors
import py3Dmol
from datetime import datetime
from PIL import Image
import io
import json
import base64
from chat_storage import ChatSessionStorage
import os
import logging
from typing import Dict, Any, Union, List
from simulate_ai import (
    process_smiles_in_text,
    get_chemistry_lab,
    process_smiles,
    process_smiles_for_3d,
    llava_call,
    llava_config_list,
    tavily_search,
    process_search_results,
    is_valid_url,
    MoleculeValidator
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

chat_storage = ChatSessionStorage()
chat_sessions = chat_storage.chat_sessions

def create_app():
    app = Flask(__name__, static_folder='static')
    CORS(app)
    socketio = SocketIO(app)

    # Initialize ChemistryLab
    chemistry_lab = None
    literature_path = ""
    web_url_path = ""
    molecule_validator = MoleculeValidator() 

    @app.route('/get_molecule_details', methods=['POST'])
    def get_molecule_details():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            # Use molecule_validator instead of creating new instance
            molecule_data = molecule_validator.process_smiles(smiles)
            if molecule_data is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            mol = Chem.MolFromSmiles(molecule_data["smiles"]) 
            if mol is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            conf = mol.GetConformer()
            
            # Add atoms information
            atoms = []
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                atoms.append({
                    "elem": atom.GetSymbol(),
                    "x": pos.x, 
                    "y": pos.y,
                    "z": pos.z
                })
                
            molecule_data["atoms"] = atoms
            molecule_data["pdb"] = Chem.MolToPDBBlock(mol)

            return jsonify(molecule_data)

        except Exception as e:
            logger.error(f"Error getting molecule details: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/get_molecule_info', methods=['POST'])
    def get_molecule_info():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            # Get detailed molecular information
            info = {
                "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.ExactMolWt(mol), 2),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "charge": Chem.GetFormalCharge(mol),
                "logP": round(Descriptors.MolLogP(mol), 2),
                "TPSA": round(Descriptors.TPSA(mol), 2),
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol)
            }
            return jsonify(info)

        except Exception as e:
            logger.error(f"Error getting molecule info: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/process_smiles', methods=['POST'])
    def process_smiles():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            result = process_smiles(smiles)
            if result is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            return jsonify(result)

        except Exception as e:
            logger.error(f"Error processing SMILES: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/render_3d_structure', methods=['POST'])
    def render_3d_structure():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            structure = process_smiles_for_3d(smiles)
            if structure is None:
                return jsonify({"error": "Failed to generate 3D structure"}), 400

            return jsonify({"structure": structure})

        except Exception as e:
            logger.error(f"Error rendering 3D structure: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/generate_molecule_image', methods=['GET'])
    def generate_molecule_image():
        smiles = request.args.get('smiles')
        if not smiles:
            return jsonify({"error": "No SMILES provided"}), 400

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            img = Draw.MolToImage(mol)
            img_io = io.BytesIO()
            img.save(img_io, 'PNG')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')
        except Exception as e:
            logger.error(f"Error generating molecule image: {str(e)}")
            return jsonify({"error": str(e)}), 400

    @app.route('/configure', methods=['POST'])
    def configure():
        nonlocal chemistry_lab, literature_path, web_url_path
        data = request.json
        literature_path = data.get('literature_path', '')
        web_url_path = data.get('web_url_path', '')
        chemistry_lab = get_chemistry_lab(literature_path)
        logger.info(f"Configured with literature_path: {literature_path}, web_url_path: {web_url_path}")
        return jsonify({'status': 'Configuration updated', 'literature_path': literature_path})

    @app.route('/simulate', methods=['POST'])
    def simulate():
        nonlocal chemistry_lab, web_url_path, literature_path
        if not chemistry_lab:
            chemistry_lab = get_chemistry_lab(literature_path)

        user_input = request.form.get('message', '')
        image_file = request.files.get('image')
        new_literature_path = request.form.get('literature_path', '')
        new_web_url_path = request.form.get('web_url_path', '')
        session_id = request.form.get('session_id', 'default')

        logger.info(f"Received request - User input: {user_input}, Literature path: {new_literature_path}, Web URL path: {new_web_url_path}")

        # Update paths if necessary 
        if new_literature_path and new_literature_path != literature_path:
            logger.info(f"Updating literature path from {literature_path} to {new_literature_path}")
            literature_path = new_literature_path
            chemistry_lab = get_chemistry_lab(literature_path)

        web_url_path = new_web_url_path

        try:
            # Initialize response components
            search_results = []
            llava_response = None
            image_data = None 
            image_bytes = None

            # Process image if provided
            if image_file:
                try:
                    image_bytes = image_file.read()
                    image_data = base64.b64encode(image_bytes).decode('utf-8')
                    llava_response = llava_call(user_input, image_bytes, llava_config_list[0])
                    logger.info(f"LLaVA Response: {llava_response}")

                    # Perform web search with combined query
                    combined_query = f"{user_input} {llava_response}"
                    search_results = tavily_search(combined_query, url=web_url_path) if web_url_path and is_valid_url(web_url_path) else tavily_search(combined_query)
                except Exception as e:
                    logger.error(f"Error in image processing: {str(e)}")
                    llava_response = f"Error processing image: {str(e)}"
            else:
                # Perform web search with original query
                try:
                    search_results = tavily_search(user_input, url=web_url_path) if web_url_path and is_valid_url(web_url_path) else tavily_search(user_input)
                except Exception as e:
                    logger.error(f"Error in web search: {str(e)}")

            # Perform RAG search if database exists
            rag_results = None
            if chemistry_lab.db:
                try:
                    rag_results = rag_search(user_input)
                except Exception as e:
                    logger.error(f"Error in RAG search: {str(e)}")

            # Process user input and simulate
            response = chemistry_lab.process_user_input(
                user_input,
                image_data=image_bytes,
                literature_path=literature_path,
                web_url_path=web_url_path
            )

            chemistry_lab.simulate(1)
            performance_analysis = chemistry_lab.analyze_system_performance()
            
            # Update agent information
            updated_agents = [
                {
                    'name': agent.name,
                    'evolutionLevel': agent.evolution_level,
                    'skills': list(agent.skills)
                }
                for agent in chemistry_lab.agents
            ]

            # Process and enhance response messages
            first_assistant_found = False
            feedback_data = {}
            
            for msg in response:
                if msg.get('role') == 'assistant':
                    # Process feedback if available
                    if msg.get('feedback_rating'):
                        feedback_data[msg.get('name', 'unknown')] = {
                            'rating': msg['feedback_rating'],
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Enhance first assistant message with additional info
                    if not first_assistant_found:
                        first_assistant_found = True
                        
                        # Add image analysis if available
                        if llava_response:
                            msg['content'] = f"Image Analysis: {llava_response}\n\n{msg['content']}"
                        
                        # Process and add search results
                        processed_results = []
                        
                        if search_results:
                            web_results = process_search_results(search_results)
                            if web_results:
                                processed_results.append({
                                    'type': 'web',
                                    'results': web_results
                                })

                        if rag_results:
                            rag_processed = process_search_results([{
                                'content': rag_results,
                                'title': 'Literature Search Result',
                                'url': None
                            }])
                            if rag_processed:
                                processed_results.append({
                                    'type': 'rag',
                                    'results': rag_processed
                                })

                        if processed_results:
                            msg['search_results'] = processed_results

            # Add performance analysis to response
            response.append({
                'role': 'system',
                'name': 'Performance_Analysis',
                'content': json.dumps({
                    'performance_analysis': performance_analysis,
                    'updated_agents': updated_agents
                })
            })

            # Create history entry
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_input': user_input,
                'image_data': image_data,
                'literature_path': literature_path,
                'web_url_path': web_url_path,
                'response': response,
                'feedback': feedback_data,
                'files': {
                    'image': bool(image_data),
                    'literature': bool(literature_path),
                    'web_url': bool(web_url_path)
                },
                'performance_metrics': {
                    'response_time': time.time() - request.start_time if hasattr(request, 'start_time') else None,
                    'agent_updates': len(updated_agents),
                    'search_results_count': len(search_results) if search_results else 0
                }
            }

            # Store history
            try:
                chat_storage.add_session_entry(session_id, history_entry)
            except Exception as storage_error:
                logger.error(f"Error storing chat history: {str(storage_error)}")

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}", exc_info=True)
            error_response = [{
                'role': 'assistant',
                'name': 'System',
                'content': f"Error processing your input: {str(e)}"
            }]
            
            # Create error history entry
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'user_input': user_input,
                'image_data': image_data,
                'literature_path': literature_path,
                'web_url_path': web_url_path,
                'response': error_response,
                'feedback': {},
                'files': {
                    'image': bool(image_data),
                    'literature': bool(literature_path),
                    'web_url': bool(web_url_path)
                },
                'error': {
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            }
            
            # Store error history
            try:
                chat_storage.add_session_entry(session_id, error_entry)
            except Exception as storage_error:
                logger.error(f"Error storing error history: {str(storage_error)}")
                
            return jsonify(error_response), 500

    @app.route('/render_3d_molecule', methods=['POST'])
    def render_3d_molecule():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            mol = process_smiles_for_3d(smiles)
            if mol is None:
                return jsonify({"error": "Failed to process SMILES"}), 400

            return jsonify(mol)
        except Exception as e:
            logger.error(f"Error rendering 3D molecule: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/initialize', methods=['POST'])
    def initialize_chat():
        nonlocal chemistry_lab
        chemistry_lab = get_chemistry_lab()
        return jsonify({'status': 'Chemistry Lab initialized successfully'})

    @app.route('/feedback', methods=['POST'])
    def feedback():
        try:
            feedback_data = request.json
            if not feedback_data:
                return jsonify({'error': 'No feedback data provided'}), 400

            # Get session ID from URL parameters, default to 'default' if not provided 
            session_id = request.args.get('session_id', 'default')

            # Get message index from URL parameters, default to -1 if not provided
            try:
                message_index = int(request.args.get('message_index', -1))
            except ValueError:
                return jsonify({'error': 'Invalid message index'}), 400

            # Initialize ChatSessionStorage
            chat_storage = ChatSessionStorage()

            # Add feedback using the chat storage method
            success = chat_storage.add_feedback(session_id, message_index, feedback_data)

            if success:
                return jsonify({'status': 'Feedback received and stored successfully'})
            else:
                return jsonify({'error': 'Failed to save feedback'}), 400

        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return jsonify({'error': str(e)}), 400

    @app.route('/history', methods=['GET'])
    def history():
        try:
            session_id = request.args.get('session_id', 'default')
            history_data = chat_storage.get_session_history(session_id)
            
            # Process history data to include feedback
            if session_id == 'all':
                for session in history_data['sessions']:
                    for msg in session['response']:
                        if msg.get('role') == 'assistant' and session.get('feedback'):
                            agent_name = msg.get('name')
                            if agent_name in session['feedback']:
                                msg['feedback_rating'] = session['feedback'][agent_name]
            else:
                for session in history_data['session']:
                    for msg in session['response']:
                        if msg.get('role') == 'assistant' and session.get('feedback'):
                            agent_name = msg.get('name')
                            if agent_name in session['feedback']:
                                msg['feedback_rating'] = session['feedback'][agent_name]
            
            return jsonify(history_data)
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
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

    @app.route('/get_3d_structure', methods=['POST'])
    def get_3d_structure():
        try:
            data = request.json
            smiles = data.get('smiles')
            if not smiles:
                return jsonify({"error": "No SMILES provided"}), 400

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return jsonify({"error": "Invalid SMILES string"}), 400

            # Generate 3D conformation
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

            # Convert to JSON structure data
            conf = mol.GetConformer()
            structure = {
                "atoms": [
                    {
                        "serial": i + 1,
                        "elem": atom.GetSymbol(),
                        "x": conf.GetAtomPosition(i).x,
                        "y": conf.GetAtomPosition(i).y,
                        "z": conf.GetAtomPosition(i).z
                    }
                    for i, atom in enumerate(mol.GetAtoms())
                ],
                "bonds": [
                    {
                        "start": bond.GetBeginAtomIdx() + 1,
                        "end": bond.GetEndAtomIdx() + 1,
                        "order": int(bond.GetBondTypeAsDouble())
                    }
                    for bond in mol.GetBonds()
                ]
            }

            return jsonify({"structure": structure})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # WebSocket event handlers
    @socketio.on('connect')
    def handle_connect():
        logger.info('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('Client disconnected')

    return app

if __name__ == '__main__':
    app = create_app()
    socketio = SocketIO(app)
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
