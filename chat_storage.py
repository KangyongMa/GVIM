import json
import os
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSessionStorage:
    def __init__(self, storage_dir: str = 'chat_history') -> None:
        """
        Initialize chat session storage
        
        Args:
            storage_dir (str): Directory to store chat history files
        """
        self.storage_dir = storage_dir
        self.sessions_file = os.path.join(storage_dir, 'sessions.json')
        self.ensure_storage_dir()
        self.chat_sessions = self.load_sessions()
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        feedback_analysis = {
            'agent_ratings': {},
            'topic_ratings': {},
            'improvement_areas': []
        }
        
        for session in self.chat_sessions['session_history']:
            if 'feedback' in session:
                for agent, rating in session.get('feedback', {}).items():
                    if agent not in feedback_analysis['agent_ratings']:
                        feedback_analysis['agent_ratings'][agent] = []
                    feedback_analysis['agent_ratings'][agent].append(rating)
                
                topic = self.extract_topic(session['user_input'])
                if topic not in feedback_analysis['topic_ratings']:
                    feedback_analysis['topic_ratings'][topic] = []
                feedback_analysis['topic_ratings'][topic].extend(session['feedback'].values())
    
        for agent, ratings in feedback_analysis['agent_ratings'].items():
            if len(ratings) >= 2:
                trend = np.polyfit(range(len(ratings)), ratings, 1)[0]
                if trend < 0:
                    feedback_analysis['improvement_areas'].append({
                        'agent': agent,
                        'trend': trend
                    })
                    
        return feedback_analysis

    def ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists"""
        try:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
                logger.info(f"Created storage directory: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
            raise

    def load_sessions(self) -> Dict[str, Any]:
        """
        Load sessions from file
        
        Returns:
            Dict containing session data
        """
        default_sessions = {
            'session_history': [],
            'active_sessions': {}
        }
        
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    sessions = json.load(f)
                logger.info(f"Successfully loaded sessions from {self.sessions_file}")
                return sessions
            except Exception as e:
                logger.error(f"Error loading sessions from {self.sessions_file}: {e}")
                return default_sessions
        else:
            logger.info(f"No existing sessions file found at {self.sessions_file}")
            return default_sessions

    def save_sessions(self) -> None:
        """Save sessions to file"""
        try:
            # Ensure the storage directory exists before saving
            self.ensure_storage_dir()
            
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_sessions, f, ensure_ascii=False, indent=2)
            logger.info(f"Successfully saved sessions to {self.sessions_file}")
        except Exception as e:
            logger.error(f"Error saving sessions to {self.sessions_file}: {e}")
            raise

    def add_session_entry(self, session_id: str, entry: Dict[str, Any]) -> None:
        """
        Add a new session entry
        
        Args:
            session_id (str): Session identifier
            entry (Dict): Session entry data
        """
        try:
            # Ensure required dictionaries exist
            if 'active_sessions' not in self.chat_sessions:
                self.chat_sessions['active_sessions'] = {}
            if 'session_history' not in self.chat_sessions:
                self.chat_sessions['session_history'] = []
            
            # Initialize session list if needed
            if session_id not in self.chat_sessions['active_sessions']:
                self.chat_sessions['active_sessions'][session_id] = []
            
            # Add entry to both active sessions and history
            self.chat_sessions['active_sessions'][session_id].append(entry)
            self.chat_sessions['session_history'].append(entry)
            
            # Save after adding new entry
            self.save_sessions()
            logger.info(f"Added new session entry for session {session_id}")
        except Exception as e:
            logger.error(f"Error adding session entry: {e}")
            raise

    def get_session_history(self, session_id: str = 'all') -> Dict[str, Any]:
        """
        Get session history
        
        Args:
            session_id (str): Session identifier or 'all' for complete history
            
        Returns:
            Dict containing session history
        """
        try:
            if session_id == 'all':
                # Return all historical sessions with proper formatting
                formatted_sessions = []
                for session in self.chat_sessions['session_history']:
                    formatted_session = {
                        'timestamp': session['timestamp'],
                        'user_input': session['user_input'],
                        'image_data': session.get('image_data'),
                        'literature_path': session.get('literature_path', ''),
                        'web_url_path': session.get('web_url_path', ''),
                        'response': session['response'],
                        'feedback': session.get('feedback', {}),
                        'files': session['files']
                    }
                    formatted_sessions.append(formatted_session)
                
                return {
                    'sessions': formatted_sessions,
                    'count': len(formatted_sessions)
                }
            
            # Return specific session history
            session_history = self.chat_sessions['active_sessions'].get(session_id, [])
            formatted_history = []
            
            for session in session_history:
                formatted_session = {
                    'timestamp': session['timestamp'],
                    'user_input': session['user_input'],
                    'image_data': session.get('image_data'),
                    'literature_path': session.get('literature_path', ''),
                    'web_url_path': session.get('web_url_path', ''),
                    'response': session['response'],
                    'feedback': session.get('feedback', {}),
                    'files': session['files']
                }
                formatted_history.append(formatted_session)
            
            return {
                'session': formatted_history,
                'count': len(formatted_history)
            }
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return {'sessions': [], 'count': 0}

    def add_feedback(self, session_id: str, message_index: int, feedback_data: Dict[str, Any]) -> bool:
        try:
            # Ensure the session exists
            if 'active_sessions' not in self.chat_sessions:
                self.chat_sessions['active_sessions'] = {}
                
            if session_id not in self.chat_sessions['active_sessions']:
                self.chat_sessions['active_sessions'][session_id] = []

            # Get the session messages
            sessions = self.chat_sessions['active_sessions'][session_id]
            
            # If message_index is -1, add feedback to the last message
            if message_index == -1:
                message_index = len(sessions) - 1
                
            if 0 <= message_index < len(sessions):
                # Add or update feedback
                if 'feedback' not in sessions[message_index]:
                    sessions[message_index]['feedback'] = {}
                
                # Store feedback for each agent
                for agent_name, rating in feedback_data.items():
                    sessions[message_index]['feedback'][agent_name] = rating
                
                # Also update the corresponding entry in session_history
                history_index = self.find_history_index(session_id, message_index)
                if history_index is not None:
                    if 'feedback' not in self.chat_sessions['session_history'][history_index]:
                        self.chat_sessions['session_history'][history_index]['feedback'] = {}
                    self.chat_sessions['session_history'][history_index]['feedback'].update(feedback_data)
                
                # Save changes
                self.save_sessions()
                logger.info(f"Added feedback for session {session_id}, message {message_index}")
                return True
                
            logger.warning(f"Invalid message index {message_index} for session {session_id}")
            return False
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            return False

    def find_history_index(self, session_id: str, message_index: int) -> Optional[int]:
        """
        Find the corresponding index in session_history for a message
        
        Args:
            session_id (str): Session identifier
            message_index (int): Index in active_sessions
            
        Returns:
            Optional[int]: Index in session_history or None if not found
        """
        try:
            if message_index < 0:
                return None
                
            active_session = self.chat_sessions['active_sessions'].get(session_id, [])
            if message_index >= len(active_session):
                return None
                
            target_message = active_session[message_index]
            
            # Look for matching timestamp and content in session_history
            for i, history_entry in enumerate(self.chat_sessions['session_history']):
                if (history_entry['timestamp'] == target_message['timestamp'] and
                    history_entry['user_input'] == target_message['user_input']):
                    return i
                    
            return None
        except Exception as e:
            logger.error(f"Error finding history index: {e}")
            return None

    def clear_sessions(self) -> None:
        """Clear all sessions and save the empty state"""
        try:
            self.chat_sessions = {
                'session_history': [],
                'active_sessions': {}
            }
            self.save_sessions()
            logger.info("Cleared all sessions")
        except Exception as e:
            logger.error(f"Error clearing sessions: {e}")
            raise