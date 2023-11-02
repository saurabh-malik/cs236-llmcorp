# Corporate LLM Integration Project

## Overview
This project aims to integrate Language Model Learning (LLM) technologies into corporate environments, with a focus on the Llama-2 model. The integration addresses key challenges such as data confidentiality, model hallucinations, and the need for domain-specific responses. Using a dual-strategy approach, we combine the Retrieval-Augmented Generation (RAG) method with fine-tuning on corporate-specific data to ensure accuracy, data security, and reliable model responses.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python (version 3.8 or later)
- pip (Python package manager)

### Installation

1. **Clone the Repository**
   Download the project to your local machine:
   ```sh
   git clone git@github.com:saurabh-malik/cs236-llmcorp.git
   cd cs236-llmcorp
2. **Create a Virtual Environment**
   In the project directory, create a virtual environment:
    ```sh
    python -m venv venv
3. **Activate the Virtual Environment**
   - On Windows:
   ```sh
   .\venv\Scripts\activate
   ```
   - On Unix or MacOS:
   ```sh
   source venv/bin/activate
   ```
4. **Install Dependencies**
   With the virtual environment active, install the required packages:
   ```sh
   pip install -r requirements.txt

### Usage
1. **Initialize the FAISS Vector Database**\
   Index your PDF data to create a local FAISS vector database instance:
   ```sh
   python app/indexer.py
2. **Launch the Corporate Agent**\
Start the local Llama-2 model instance and the corporate agent (chat agent):
   ```sh
   python app/corporate_agent.py
3. **Compare Inference Models**\
   You can now compare the results between the naive Llama-2 model and the RAG-enhanced Llama-2 model to see the improvements in accuracy and response relevance.\
   **Query:** "Whos is the author of '2D excitation information by MPS method on infinite helixes' paper?"
   ```sh
   -----------------Response from agent powered by Naive LLM--------------------
   The paper "2D excitation information by MPS method on infinite helixes" was written by Y. C. Kim and S. K. Lee.

   -----------------Response from agent powered by RAG-enhanced Llama-2 model--------------------
   The author of the paper is Xing-Yu Zhang, Runze Chi, Yang Liu, and Lei Wang.

   ```
## Contributing

If you're interested in contributing, please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b {firstname.lastname}-featureName`)
3. Make your changes.
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin {firstname.lastname}-featureName`)
6. Open a Pull Request


