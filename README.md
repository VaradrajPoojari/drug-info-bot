# Medical Drug Query and Summarization Prototype

## Overview
This repository contains the codebase for a prototype tool designed to answer medical drug queries and summarize relevant information using a Large Language Model (LLM). The tool integrates various components, including a Knowledge Base, LLM, User Interface (UI), and containerization for ease of deployment.

## Setup
1. Create a `.env` file in the root directory with the following environment variables:
PINECONE_API_KEY=<your_pinecone_api_key>
OPENAI_API_KEY=<your_openai_api_key>
X-API-KEY=<your_google_serper_api_key>

2. Build and run the Docker container:
```bash
docker build -t medical-drug-query .
docker run -p 8501:8501 medical-drug-query
```
## Components
Knowledge Base: Utilizes a vector database with drug information, supplemented by data from Google Search SERP API and Wikipedia Search API.
LLM (Large Language Model): Integrates OpenAI GPT-4 model for content filtering, ranking, and summarization.
UI (User Interface): Built using the Streamlit package, offering a simple query input and submission interface.
Containerization: Docker containerizes the application for easy deployment and execution.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
