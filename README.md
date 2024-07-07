# WebLINX Chatbot - COMP 545 Final Project
## Natural Language Understanding with Deep Learning - Application Track

### Project Overview

The WebLINX Chatbot is an advanced conversational interface developed as a group project for COMP 545, which focuses on Natural Language Understanding with Deep Learning. This chatbot enhances human-computer interaction by enabling users to navigate and interact with websites through a conversational interface. Employing the latest advancements in NLP, our system allows users to execute web-based tasks such as clicking buttons, scrolling through pages, filling out forms, and seamlessly navigating across websites.

### Features

- **Multi-Turn Dialogue**: Engage in a dynamic conversation with the chatbot to perform tasks on the web.
- **Advanced NLP Capabilities**: Utilizes a fine-tuned LLaMA-2 7B model for understanding and processing user commands.
- **Automated Web Interaction**: Translates the model's structured output into Selenium-compatible actions.
- **User-Friendly Interface**: Streamlit-based front end for easy interaction with the chatbot.

### Prerequisites

- An Anaconda or Miniconda distribution of Python 3.9 or higher installed on your machine.
- Access to the internet to fetch dependencies and interact with remote APIs.

### Installation

1. **Clone the Repository:**
```
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

2. **Set Up Environment Variables:**
Create a `.env` file in the project root and fill in the Hugging Face API endpoints:
```
API_URL_DMR="YOUR_DMR_API_KEY"
API_URL_ACTION="YOUR_ACTION_LLM_KEY"
```

3. **Create a Conda Environment:**
```
conda create -n weblinx_chatbot python=3.9
conda activate weblinx_chatbot
```

4. **Install Dependencies:**
```
pip install -r requirements.txt
```

### Running the Application
To start the WebLINX Chatbot, run the following command in your terminal:
```
streamlit run main.py
```

Navigate to the provided local URL in your web browser to interact with the chatbot interface.

### How to Use
* Type your commands into the chat input box to interact with websites.
* Use the "Continue" command to proceed with actions or "Quit" to terminate the session.
* View the interaction history and web navigation results in real-time through the Streamlit interface.

### Troubleshooting
* If you encounter issues with network connections or API limits, verify your `.env` file settings and ensure your network is stable.
* For any unexpected behavior or errors, try restarting Streamlit or reactivating your Conda environment.
