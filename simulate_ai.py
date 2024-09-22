import json
import os
import random
import re
import base64
import io
import warnings
import logging
from collections import deque
from typing import List, Dict, Any, Union
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.llava_agent import LLaVAAgent
import replicate
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.agents import Tool
from tavily import TavilyClient
from rdkit import Chem
from rdkit.Chem import Draw
from tenacity import retry, stop_after_attempt, wait_fixed

warnings.filterwarnings("ignore")
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TAVILY_API_KEY"] = "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"
os.environ["REPLICATE_API_TOKEN"] = ""  # Replace with your actual token

config_list = [
    {
        "model": "llama3-8b-8192",
        "api_key": "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",
        "base_url": "https://api.groq.com/openai/v1"
    },
    {
        "model": "MistralNemo",
        "api_key": "NA",
        "base_url": "https://77bc-34-125-119-159.ngrok-free.app/v1"
    },
    {
        "model": "mixtral-8x7b-32768",
        "api_key": "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",
        "base_url": "https://api.groq.com/openai/v1"
    },
]

llava_config_list = [
    {
        "model": "whatever, will be ignored for remote",
        "api_key": "None",
        "base_url": "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    }
]

def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def llava_call(prompt: str, image_data: Union[bytes, None] = None, config: Dict[str, Any] = None) -> str:
    if config is None:
        config = llava_config_list[0]

    base_url = config["base_url"]

    inputs = {
        "prompt": prompt,
    }

    if image_data:
        # Convert image data to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        inputs["image"] = f"data:image/jpeg;base64,{image_base64}"

    try:
        output = replicate.run(base_url, input=inputs)
        return "".join(output)
    except Exception as e:
        logger.error(f"Error calling LLaVA API: {str(e)}")
        return f"Error processing image: {str(e)}"

def load_documents(file_path):
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return []

    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()

# Load experiment data
experiment_data = load_documents("E://HuaweiMoveData//Users//makangyong//Desktop//output.txt")

# Load literature (if available)
literature_path = ""  # Update this path if you have literature to load
literature = load_documents(literature_path) if literature_path else []

# Combine all documents
all_documents = experiment_data + literature

# Check if we have any documents before proceeding
if not all_documents:
    logger.warning("No documents loaded. Skipping embedding and vector store creation.")
    db = None
else:
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(texts, embeddings)

llm = ChatOpenAI(model_name="gemma2-9b-it", openai_api_key=config_list[0]["api_key"], openai_api_base=config_list[0]["base_url"])
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

# Initialize Tavily client with error handling and fallback
def fallback_search(query):
    logger.warning(f"Fallback search used for query: {query}")
    return f"Fallback search result for: {query}"

try:
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
except Exception as e:
    logger.error(f"Error initializing Tavily client: {str(e)}")
    tavily_client = None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def tavily_search(query, url=None):
    try:
        if tavily_client is None:
            return fallback_search(query)

        if url and is_valid_url(url):
            response = tavily_client.search(query=f"site:{url} {query}")
        else:
            response = tavily_client.search(query=query)
        
        logger.info(f"Tavily search performed for query: {query}")
        results = [{"url": obj["url"], "content": obj["content"]} for obj in response["results"]]
        return results
    except Exception as e:
        logger.error(f"Error performing Tavily search: {str(e)}")
        return fallback_search(query)

tavily_tool = Tool(
    name="Tavily Search",
    func=tavily_search,
    description="Useful for searching the internet for recent information on Chemistry."
)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def rag_search(query):
    try:
        return rag_chain.invoke({"query": query})
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        return f"Error performing RAG search: {str(e)}"

agent_llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234,
    "functions": [
        {
            "name": "rag_search",
            "description": "Search in the loaded chemical documents using RAG",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the chemical documents"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "tavily_search",
            "description": "Search the internet for chemical information using Tavily",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for chemical information on the internet"
                    }
                },
                "required": ["query"]
            }
        }
    ]
}

manager_llm_config = {
    "config_list": config_list,
    "timeout": 60,
    "temperature": 0.8,
    "seed": 1234
}

def visualize_smiles(smiles: str) -> str:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return f"Invalid SMILES string: {smiles}"

        img = Draw.MolToImage(molecule)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        return f"Error generating molecule image: {str(e)}"

def process_smiles_in_text(text: str) -> str:
    smiles_pattern = r'\b[CN]\S+'

    def replace_with_image(match):
        smiles = match.group(0)
        try:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                return smiles  # Return original text if not a valid SMILES

            if not os.path.exists('static/images'):
                os.makedirs('static/images')

            filename = f"molecule_{hash(smiles)}.png"
            filepath = os.path.join('static/images', filename)

            img = Draw.MolToImage(molecule)
            img.save(filepath)

            return f"{smiles}\n<img src='/static/images/{filename}' alt='Molecule' class='molecule-image'>"
        except Exception as e:
            logger.error(f"Error processing SMILES string: {str(e)}")
            return smiles  # Return original text if there's an error

    return re.sub(smiles_pattern, replace_with_image, text)

class ChemistryAgent(autogen.AssistantAgent):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.knowledge_base = set()
        self.skills = set()
        self.performance_history = deque(maxlen=10)
        self.interaction_history = []

    def learn(self, new_knowledge):
        self.knowledge_base.add(new_knowledge)
        logger.info(f"{self.name} learned: {new_knowledge}")

    def acquire_skill(self, new_skill):
        self.skills.add(new_skill)
        logger.info(f"{self.name} acquired new skill: {new_skill}")

    def evaluate_performance(self, user_feedback):
        # Convert user feedback to a numerical score
        score = self.feedback_to_score(user_feedback)
        self.performance_history.append(score)
        logger.info(f"{self.name}'s performance score: {score}")

    def feedback_to_score(self, feedback):
        # Simple mapping of feedback to scores
        feedback_scores = {
            "excellent": 1.0,
            "good": 0.8,
            "average": 0.6,
            "poor": 0.4,
            "very poor": 0.2
        }
        return feedback_scores.get(feedback.lower(), 0.5)  # Default to average if unknown

    def evolve(self):
        avg_performance = sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0
        if avg_performance > 0.7:
            self.improve()
        elif avg_performance < 0.5:
            self.refine_skills()

    def improve(self):
        # Analyze recent interactions to identify areas for improvement
        common_topics = self.analyze_interactions()
        new_skill = f"Advanced_{random.choice(common_topics)}"
        self.acquire_skill(new_skill)
        logger.info(f"{self.name} evolved and gained a new skill: {new_skill}")

    def refine_skills(self):
        # Identify the least used skill and replace it
        if self.skills:
            least_used_skill = min(self.skills, key=lambda s: self.skill_usage_count(s))
            self.skills.remove(least_used_skill)
            new_skill = f"Refined_{least_used_skill}"
            self.acquire_skill(new_skill)
            logger.info(f"{self.name} refined skills: Removed {least_used_skill}, Added {new_skill}")

    def analyze_interactions(self):
        # Simple analysis of recent interactions to identify common topics
        topics = [interaction['topic'] for interaction in self.interaction_history[-10:]]
        return list(set(topics))  # Return unique topics

    def skill_usage_count(self, skill):
        # Count how many times a skill was used in recent interactions
        return sum(1 for interaction in self.interaction_history[-20:] if skill in interaction['skills_used'])

    def process_user_input(self, user_input, image_data=None):
        response = super().process_user_input(user_input)

        if image_data:
            llava_response = llava_call(user_input, image_data, llava_config_list[0])
            response = f"Image analysis: {llava_response}\n\n{response}"

        topic = self.extract_topic(user_input)
        skills_used = self.identify_skills_used(user_input, response)

        self.interaction_history.append({
            'user_input': user_input,
            'response': response,
            'topic': topic,
            'skills_used': skills_used
        })

        return response

    def extract_topic(self, text):
        # Simple topic extraction (can be improved with NLP techniques)
        keywords = ['reaction', 'compound', 'element', 'analysis', 'safety']
        for keyword in keywords:
            if keyword in text.lower():
                return keyword
        return 'general chemistry'

    def identify_skills_used(self, user_input, response):
        # Identify skills used in the interaction (can be improved with more sophisticated analysis)
        skills_used = []
        for skill in self.skills:
            if skill.lower() in user_input.lower() or skill.lower() in response.lower():
                skills_used.append(skill)
        return skills_used

    def send(self, message: Union[str, Dict[str, Any]], recipient: autogen.Agent, request_reply: bool = None, silent: bool = False) -> None:
        if isinstance(message, str):
            message = process_smiles_in_text(message)
        elif isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = process_smiles_in_text(message["content"])
        super().send(message, recipient, request_reply, silent)

class ChemistryLab:
    def __init__(self, literature_path=""):
        self.agents = []
        self.groupchat = None
        self.manager = None
        self.literature_path = literature_path
        self.setup_agents()
        self.load_documents()

    def load_documents(self):
        # Load experiment data
        experiment_data = load_documents("E://HuaweiMoveData//Users//makangyong//Desktop//output.txt")

        # Load literature (if available)
        literature = load_documents(self.literature_path) if self.literature_path else []

        # Combine all documents
        all_documents = experiment_data + literature

        # Check if we have any documents before proceeding
        if not all_documents:
            logger.warning("No documents loaded. Skipping embedding and vector store creation.")
            self.db = None
        else:
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(all_documents)

            embeddings = HuggingFaceEmbeddings()
            self.db = Chroma.from_documents(texts, embeddings)

    def setup_agents(self):
        agent_configs = [
            ("Lab_Director", "You are the director of a chemistry laboratory. Assign tasks, ask questions about chemical experiments, and oversee the research process."),
            ("Senior_Chemist", "You are a senior chemist with expertise in organic, inorganic, and physical chemistry. Provide detailed answers and insights on complex chemical processes."),
            ("Lab_Manager", "You are a laboratory manager responsible for overseeing chemical experiments, ensuring safety protocols, and managing resources. Plan and design projects with efficiency and safety in mind."),
            ("Safety_Officer", "You are a chemical safety officer responsible for reviewing experimental procedures and ensuring compliance with safety regulations. Provide feedback on safety measures and potential hazards."),
            ("Analytical_Chemist", "You are an analytical chemist specializing in chemical analysis techniques and instrumentation. Provide expertise on analytical methods, data interpretation, and quality control.")
        ]

        for name, system_message in agent_configs:
            agent = ChemistryAgent(name=name, system_message=system_message, llm_config=agent_llm_config)
            self.agents.append(agent)
            agent.register_function(
                function_map={
                    "rag_search": rag_search,
                    "tavily_search": tavily_search
                }
            )

    def setup_groupchat(self):
        self.groupchat = autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False,
        )
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=manager_llm_config)

    def process_user_input(self, user_input, image_data=None, literature_path=None, web_url_path=None):
        logger.info(f"Processing user input: {user_input}")
        logger.info(f"Web URL Path: {web_url_path}")

        if not self.groupchat or not self.manager:
            self.setup_groupchat()

        try:
            if image_data:
                llava_response = llava_call(user_input, image_data, llava_config_list[0])
                user_input = f"{user_input}\n[IMAGE_ANALYSIS:{llava_response}]"

            # Check if a valid URL is provided
            if web_url_path and is_valid_url(web_url_path):
                search_result = tavily_search(user_input, url=web_url_path)
            else:
                search_result = tavily_search(user_input)

            user_input = f"{user_input}\n[WEB_SEARCH:{search_result}]"

            chat_result = self.manager.initiate_chat(
                self.agents[0],
                message=user_input,
            )

            chat_history = chat_result.chat_history if hasattr(chat_result, 'chat_history') else chat_result

            processed_messages = []
            for message in chat_history:
                logger.info(f"Processing message: {message}")
                if isinstance(message, dict) and 'role' in message:
                    if message['role'] == 'human':
                        processed_messages.append({
                            'role': 'user',
                            'name': 'You',
                            'content': message['content']
                        })
                    elif message['role'] == 'assistant':
                        agent_name = message.get('name', 'AI Assistant')
                        processed_content = process_smiles_in_text(message['content'])
                        processed_messages.append({
                            'role': 'assistant',
                            'name': agent_name,
                            'content': processed_content
                        })
                else:
                    logger.warning(f"Unexpected message format: {message}")

            for agent in self.agents:
                agent.evolve()

            logger.info(f"Generated responses: {json.dumps(processed_messages, indent=2)}")
            return processed_messages
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}", exc_info=True)
            return [{
                'role': 'assistant',
                'name': 'System',
                'content': f"Error processing your input: {str(e)}"
            }]

    def get_user_feedback(self, feedback):
        logger.info(f"Received feedback: {feedback}")
        for agent in self.agents:
            agent.evaluate_performance(feedback)

    def simulate(self, num_rounds):
        for round in range(num_rounds):
            logger.info(f"Starting simulation round {round + 1}")
            for agent in self.agents:
                agent.learn(f"New_ChemicalConcept_{round}")

                if random.random() > 0.7:
                    agent.acquire_skill(f"ChemicalSkill_{round}")

                performance = random.uniform(0, 1)
                agent.evaluate_performance(str(performance))  # Convert to string for compatibility

                agent.evolve()

            self.knowledge_sharing()

    def knowledge_sharing(self):
        for agent in self.agents:
            other_agents = [a for a in self.agents if a != agent]
            if other_agents:
                sharing_partner = random.choice(other_agents)
                shared_knowledge = random.choice(list(agent.knowledge_base)) if agent.knowledge_base else None
                if shared_knowledge:
                    sharing_partner.learn(shared_knowledge)
                    logger.info(f"{agent.name} shared knowledge '{shared_knowledge}' with {sharing_partner.name}")

def get_chemistry_lab(literature_path=""):
    return ChemistryLab(literature_path)

# Keep the simulate function at the end
def simulate(message, image_data=None):
    chemistry_lab = get_chemistry_lab()
    return chemistry_lab.process_user_input(message, image_data)
