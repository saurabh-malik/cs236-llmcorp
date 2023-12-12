# Corporate LLM Integration Project

## Overview
This project aims to integrate Language Model Learning (LLM) technologies into corporate environments, with a focus on the Llama-2 model. The integration addresses key challenges such as data confidentiality, model hallucinations, and the need for domain-specific responses. Using a dual-strategy approach, we combine the Retrieval-Augmented Generation (RAG) method with fine-tuning on corporate-specific data to ensure accuracy, data security, and reliable model responses.

## Getting Started

### Prerequisites
Ensure you have the following installed on your system:
- Python (version 3.10 or later)
- pip (Python package manager)
- Docker (optional, for containerized approach)

### Installation

1. **Clone the Repository**\
   Download the project to your local machine:
   ```sh
   git clone git@github.com:saurabh-malik/cs236-llmcorp.git
   cd cs236-llmcorp
2. **Create a Virtual Environment**\
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
4. **Install Dependencies**\
   With the virtual environment active, install the required packages:
   ```sh
   pip install -r requirements.txt

#### Alternative: Using Docker (Containerized Approach)
1. **Build the Docker Image**
   ```sh
   docker build -t [your-image-name] .

### Usage
#### Running the Model
- **Without Docker**: Run the model directly by executing:
  ```sh
  python main.py
- **With Docker**: Run the docker container.
  Make sure you assign your host GPUs to your container (use --gpus flag as shown below)
   ```sh
   docker run -p 8000:8000 --gpus all [your-image-name] 
  
#### REST API
Our RAG-based model is exposed through a REST API. Use the following endpoints:
1. **Inference**
 - POST `/api/v1/get_answer`
2. **Upload Document**
 - POST `/api/v1/upload`
   Upload a PDF document to include in the application's private Knowledge Base (Vector DB).
3. **Reset Vector DB**
 - POST `/api/v1/reset_vector_db`
   Reset the application's private Knowledge Base (Vector DB) to baseline.

#### Web Application
Access the web application at the root of your host URL. It provides a user-friendly interface to interact with our APIs.


### Compare Inference Models
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
 The author of the "2D excitation information by MPS method on infinite helixes" paper is Xing-Yu Zhang, Runze Chi, Yang Liu, and Lei Wang.

```
**Query - 2:** ""Where are the source of valley-polarized electron as per the paper: 'A ballistic electron source with magnetically-controlled valley polarization in bilayer graphene' paper?"
```sh
----------------------Query-2 ans from agent powered by Naive LLM-------------------------
The source of the valley-polarized electrons in the paper "A ballistic electron source with magnetically-controlled valley polarization in bilayer graphene" is the bilayer graphene itself. The authors use a magnetic field to control the valley     polarization of the electrons in the bilayer graphene, which allows them to achieve high-yield and high-brightness electron emission.
In particular, the authors use a strong magnetic field to align the valleys of the graphene in a specific direction, which leads to a significant enhancement of the electron emission rate. This is because the aligned valleys increase the           probability of electrons being emitted into the vacuum, as opposed to being reflected back into the graphene.

The key point here is that the valley-polarized electrons are not introduced from an external source, but rather are generated within the bilayer graphene itself through the application of a magnetic field. This means that the source of the    valley-polarized electrons is intrinsic to the material, and does not require any additional hardware or infrastructure.

-----------------Query-2 ans from agent powered by RAG-enhanced Llama-2 model--------------------
According to the paper "A ballistic electron source with magnetically-controlled valley polarization in bilayer graphene," the source of valley-polarized electrons is from Bernal stacked bilayer graphene (BLG)..
```
### Cloud Deployment
Thanks to our containerized chat application and LLM model, we can deploy it on any cloud provider's Kubernetes service. We have deployed it on Google Cloud's Kubernetes Engine (GKE). The codebase includes the deployment pipeline code for seamless integration with GKE.

- **Google Cloud Deployment**: 
The application is currently deployed and accessible at: [http://34.16.178.254/](http://34.16.178.254/)

## Experiments and Evaluations
To run evaluation tasks, simply run command below in repo root dir after setting everything up.
```bash
python3 -m eval.run_eval
```
In `eval/evaluation_task.py` and `eval/evaluation_task_tool.py`, you can find different evaluation tasks. Simply import and use the one you'd like to run in `run_eval.py`, and you will see the result in experiment folder.

### Fine-Tuning the Model
To further enhance our model’s accuracy and domain-specific knowledge, we fine-tuned the Llama-2-13b-chat-hf model. This process involved the following steps:

1. **Setup Virtual Environment for Fine-Tuning**:\ 
   Create a separate Virtual Environment In the project directory for llm fine-tuning.
   ```bash
   python -m venv env_finetune
   ```
2. **Actiavte the Virtual Environment for Fine-Tunning**:\
    Activate the virtual environment.
   - On Windows:
   ```sh
   .\env_finetune\Scripts\activate
   ```
   - On Unix or MacOS:
   ```sh
   source env_finetune/bin/activate
   ```
3. **Install Dependencies**:
   ```sh
   pip install -r requirements_finetune.txt
   ```
4. Data Generation for Fine-Tuning:\
   Generate fine-tuning data based on corporate public data using OpenAI models. You would need to set your OPENAI_API_KEY key as api_kep in for this step. 
   ```sh
   python scripts/finetune-data-pipeline.py generate
   ```
   This will generate pairs of Question and Answers based on provided corporate context under ./data/corporate/processed_data_multiple_qa.csv file.\
   _Note: This step uses the 'text-davinci-003' model, and OpenAI API calls are chargeable._
5. Prepare Data for LLaMA-2 Model:\
   Run the following script
   ```sh
   python scripts/finetune-data-pipeline.py finetune
   ```
   This step will prepare data as per the prompt, requeired for Llama 2 models, under ./data/corporate/finetune_data.csv file.\
   _Note: We used the following prompt format to match our RAG pipeline architecture:_
   ```plaintext
   <s>[INST] <<SYS>>
   {context}
   <</SYS>>
   Question: {User’s Question}
   [/INST]
   Helpful Answer:
   {Agent’s Answer}
   </s>
   ```
7. Run the Fine-Tuning Script:\
   Execute the fine-tuning script with the prepared data.
   ```sh
   python scripts/finetune_llm.py
   ```
   _Note: Fine-tuning configurations, such as LoRA parameters, training epochs, etc., can be set in the '**scripts/config_ft.py**' file._
## Contributing

If you're interested in contributing, please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b {firstname.lastname}-featureName`)
3. Make your changes.
4. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the Branch (`git push origin {firstname.lastname}-featureName`)
6. Open a Pull Request


