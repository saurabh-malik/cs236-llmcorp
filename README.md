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
   **Query - 1:** "Whos is the author of '2D excitation information by MPS method on infinite helixes' paper?"
   ```sh
   ----------------------Query-1 ans from agent powered by Naive Llama-2 model-------------------------
   I am trying to find the author of a paper titled "2D excitation information by MPS method on infinite helixes" but I cannot find any information about the author. Can you help me?
   The paper was published in a journal called "Journal of Physics A: Mathematical and Theoretical" in 2013, volume 46, issue 46, page 465304.
   I have tried searching online databases such as Google Scholar, ResearchGate, and Academia.edu, but I couldn't find any information about the author.
   Do you have any suggestions on how I can find the author's name?

   Thank you for your time and assistance.
   Best regards,
   [Your Name]
   
   -----------------Query-1 ans from agent powered by RAG-enhanced Llama-2 model--------------------
    The author of the paper '2D excitation information by MPS method on infinite helixes' is Xing-Yu Zhang.

   ```
   **Query - 2:** "Whos is the author of '2D excitation information by MPS method on infinite helixes' paper?"
   ```sh
   ----------------------Query-2 ans from agent powered by Naive LLM-------------------------
   The source of the valley-polarized electrons in the paper "A ballistic electron source with magnetically-controlled valley polarization in bilayer graphene" is the bilayer graphene itself. The authors use a magnetic field to control the valley     polarization of the electrons in the bilayer graphene, which allows them to achieve high-yield and high-brightness electron emission.
   In particular, the authors use a strong magnetic field to align the valleys of the graphene in a specific direction, which leads to a significant enhancement of the electron emission rate. This is because the aligned valleys increase the           probability of electrons being emitted into the vacuum, as opposed to being reflected back into the graphene.

   The key point here is that the valley-polarized electrons are not introduced from an external source, but rather are generated within the bilayer graphene itself through the application of a magnetic field. This means that the source of the    valley-polarized electrons is intrinsic to the material, and does not require any additional hardware or infrastructure.

   -----------------Query-2 ans from agent powered by RAG-enhanced Llama-2 model--------------------
   The source of valley-polarized electron is from the Bernal stacked bilayer graphene (BLG) material.


## Contributing

If you're interested in contributing, please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b {firstname.lastname}-featureName`)
3. Make your changes.
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin {firstname.lastname}-featureName`)
6. Open a Pull Request


