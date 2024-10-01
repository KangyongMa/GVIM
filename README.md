# Application of Self-Evolving AI Agents in Chemical Research: A Novel Intelligent Assistance System 🧪🤖

<div align="center">

[🧬 Paper](#) | [🔬 Blog Post](#) | [📊 Data](#) | [📹 Teaching Video](https://www.youtube.com/watch?v=78g1PUSpBNQ)

</div>

This project involves fine-tuning open-source large language models with chemical science data, evaluated using a specialized automated scoring system. The resulting chemical intelligent assistant system utilizes the fine-tuned Mistral Nemo model and can flexibly incorporate various advanced models. It integrates chemistry-specific features like molecular visualization and literature retrieval, while also possessing autonomous evolution capabilities through knowledge accumulation, skill acquisition, and collaborative mechanisms. This approach enables continuous optimization of the system's professional abilities and interaction quality, overcoming limitations of traditional static AI systems in the chemistry domain.

## 📹 Project Demo Video

Click on the image below to watch our project demonstration video:

[Watch Nature Chemistry Latest Reports](https://github.com/KangyongMa/GVIM/blob/main/Nature%20Chemistry%20latest%20reports.mp4)

## 🌟 Key Features

### Fine-tuning Large Language Models Based on Chemistry Domain Data
- Utilizing collected and curated chemistry instruction data
- Fine-tuning mainstream open-source large language models
- Developing a specialized automatic scoring system for the chemistry domain

### Innovative Chemical Intelligent Assistant System Design
- Using the fine-tuned Mistral Nemo model as one of the primary models
- Incorporating mechanisms for flexible invocation of various advanced models
- Continuously leveraging the latest AI capabilities, considering the rapid iteration of large language models

### Deep Integration of Chemistry Expertise and Requirements
- Integrating professional functions such as molecular visualization, SMILES string processing, and chemical literature retrieval
- Significantly enhancing the system's practical value in chemical research and applications

### Autonomous Evolution Capability
- Through knowledge accumulation, skill acquisition, performance evaluation, and collective collaboration mechanisms
- Continuously optimizing professional capabilities and interaction quality
- Breaking through the inherent static limitations of traditional AI systems

## 🧬 Key Additions:
1. **Configuration Section**: Added instructions for setting the file path and managing API usage limits.
2. **File Path Update**: Explicitly mentioned to update the file path based on the user's local setup.
3. **API Usage Limits**: Provided a reminder about API call limits to ensure users are aware and can plan their usage accordingly.

## API Key Setup
Ensure that you have configured the necessary API keys in your local environment before running the project. The following keys need to be manually set up:
- `TAVILY_API_KEY`
- `REPLICATE_API_TOKEN`
- `Groq_API_KEY`

## 📋 Requirements

### Installation
```bash
# Create and activate a new conda environment
conda create -n gvim python=3.9.19
conda activate gvim

# Install required packages
pip install -r requirements.txt

# Run it
python app.py
```

---

### Citation
```json
{
  "name": "Kangyong Ma",
  "affiliation": {
    "institution": "College of Physics and Electronic Information Engineering, Zhejiang Normal University",
    "city": "Jinhua City",
    "postalCode": "321000",
    "country": "China",
    "DOI": "10.26434/chemrxiv-2024-6tv8c"
  },
  "email": [
    "kangyongma@outlook.com",
    "kangyongma@gmail.com"
  ]
}
```
