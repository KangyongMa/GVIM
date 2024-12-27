# AI Agents in Chemical Research: GVIM - An Intelligent Research Assistant System 🧪🤖

> **Important Notice**: Commercial use of this project's code requires explicit authorization from the author.

<div align="center">

[![Paper](https://img.shields.io/badge/📑_Paper-Read-blue)](https://chemrxiv.org/engage/chemrxiv/article-details/66bca9acf3f4b05290da15de)
[![Install](https://img.shields.io/badge/📝_Install-Video-green)](https://www.youtube.com/watch?v=1eMwus98BB8)
[![Data](https://img.shields.io/badge/📊_Data-Access-orange)](https://huggingface.co/datasets/KANGYONGMA/GVIM)
[![Demo](https://img.shields.io/badge/🎥_Demo-Watch-red)](https://www.youtube.com/watch?v=fb8hdho_89s&t=128s)

</div>

## 📌 Overview

This project involves fine-tuning open-source large language models with chemical science data, evaluated using a specialized automated scoring system. The resulting chemical intelligent assistant system utilizes the fine-tuned large models and can flexibly integrate various advanced models. It integrates chemistry-specific features like molecular visualization and literature retrieval, while also possessing autonomous evolution capabilities through knowledge accumulation, skill acquisition, and collaborative mechanisms. This approach enables continuous optimization of the system's professional abilities and interaction quality, overcoming limitations of traditional static AI systems in the chemistry domain.

## 🌟 Key Features

### Fine-tuning Large Language Models Based on Chemistry Domain Data
- Utilizing collected and curated chemistry instruction data
- Fine-tuning mainstream open-source large language models
- Developing a specialized automatic scoring system for the chemistry domain

### Innovative Chemical Intelligent Assistant System DesignUsing the 
- Using fine-tuned models as part of the system's models
- Incorporating mechanisms for flexible invocation of various advanced models
- Continuously leveraging the latest AI capabilities, considering the rapid iteration of large language models

### Deep Integration of Chemistry Expertise and Requirements
- Integrating professional functions such as molecular visualization, SMILES string processing, and chemical literature retrieval
- Significantly enhancing the system's practical value in chemical research and applications

### Limited Improvement Capability
- Through knowledge accumulation, skill acquisition, performance evaluation, and collective collaboration mechanisms
- Continuously optimizing professional capabilities and interaction quality
- Addressing certain deficiencies of traditional systems

## 📹 Project Demonstrations

### Nature Chemistry latest reports
![Nature Chemistry latest reports](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/UI%20of%20Set%20Web%20URL.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demo%20Video%20of%20Search%20on%20the%20Official%20Website%20of%20Nature%20Chemistry%20Journal.mp4)

### Multimodal Large Model Test Demonstration Video
![Multimodal Large Model Test](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Functionality%20Expansion%E2%80%94Multimodal%20Models.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/The%20multimodal%20system%20model%20image%20recognition%20demonstration%20video..mp4)

### Multimodal system handwritten chemical formula recognition demonstration video
![Handwritten Formula Recognition](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Multimodal%20system%20handwritten%20chemical%20formula%20recognition%20demonstration%20video.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Multimodal%20system%20handwritten%20chemical%20formula%20recognition%20demonstration%20video..mp4)

### Demo 1 of an intelligent system based on local document knowledge
![Demo 1](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Demo%201%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demo%201%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.mp4)

### Demo 2 of an intelligent system based on local document knowledge
![Demo 2](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Demo%202%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demo%202%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.mp4)

### Demo 3 of an intelligent system based on local document knowledge
![Demo 3](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Demo%203%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demo%203%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.mp4)

### Demo 4 of an intelligent system based on local document knowledge
![Demo 4](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Demo%204%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demo%204%20of%20an%20intelligent%20system%20based%20on%20local%20document%20knowledge.mp4)

### Demonstration of the conversion between New Chat and History Chat
![Chat History](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/UI%20of%20System%20History.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demonstration%20of%20the%20conversion%20between%20New%20Chat%20and%20History%20Chat.mp4)

### Demonstration of Search Capabilities for Digital Discovery
![Digital Discovery](https://github.com/KangyongMa/GVIM/blob/main/Basic%20UI/Demonstration%20of%20Search%20Capabilities%20for%20Digital%20Discovery.png)
[Watch Video](https://github.com/KangyongMa/GVIM/blob/main/Demonstration%20of%20Search%20Capabilities%20for%20Digital%20Discovery.mp4)

## 🚀 Leveraging Free GPU Resources on Colab
🌟 Building on this, all the project's fine-tuned models can leverage the free GPU resources available on Colab and utilize internal network penetration technology to make calls. This is perfect for teams without funding to use this project, as it will continue to improve and optimize over time. 🚀

🎥 Here is a demonstration video on running fine-tuned models using free GPU resources: Demo Video.

## 🧬 Key Additions
1. **Configuration Section**: Added instructions for setting the file path and managing API usage limits
2. **File Path Update**: Explicitly mentioned to update the file path based on the user's local setup
3. **API Usage Limits**: Provided a reminder about API call limits to ensure users are aware and can plan their usage accordingly

## 🚀 Getting Started

### Prerequisites
```bash
# Create and activate conda environment
conda create -n gvim python=3.9.19
conda activate gvim

# Install dependencies
pip install -r requirements.txt
```

### API Configuration
Ensure that you have configured the necessary API keys in your environment:
- `TAVILY_API_KEY`
- `REPLICATE_API_TOKEN`
- `Groq_API_KEY`

### Launch
```bash
python app.py
```

## 📚 Citation

```bibtex
@article{ma2024chemai,
  author = {Kangyong Ma},
  affiliation = {College of Physics and Electronic Information Engineering, Zhejiang Normal University},
  address = {Jinhua City, 321000, China},
  doi = {10.26434/chemrxiv-2024-6tv8c},
  email = {kangyongma@outlook.com, kangyongma@gmail.com}
}
```

## 📞 Contact

For inquiries and commercial usage authorization:
- 📧 Email: kangyongma@outlook.com, kangyongma@gmail.com
- 🏢 Institution: College of Physics and Electronic Information Engineering, Zhejiang Normal University
- 📍 Location: Jinhua City, 321000, China
