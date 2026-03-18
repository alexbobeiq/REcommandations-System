# E-Commerce AI Recommendation System

A full-stack microservices application that provides real-time, personalized product recommendations based on customer purchase history. 

**Live Demo:** https://recommandations-system.onrender.com/
Being hosted on a free deployment platform, it can take some time until the services are starting after a spin down

## About The Project

This project demonstrates the integration of a Deep Learning recommendation engine with a robust Java backend and a decoupled web interface. It acts as an e-commerce platform that suggests products to users using a Neural Collaborative Filtering (NCF) approach built with PyTorch.



### Architecture & Data Flow

1. **Frontend (HTML/JS/CSS):** Hosted on Render.
2. **Backend (Java Spring Boot):** Hosted on Render (Docker container). Acts as the main API gateway, manages the product catalog using an H2 in-memory database, handles cross-origin requests (CORS), and provides fallback mechanisms.
3. **AI Engine (Python FastAPI):** Hosted on Render. Loads the pre-trained PyTorch model, computes user-item interaction scores in real-time, and serves the recommendations via a RESTful API.

## Tech Stack

* **AI & Machine Learning:** Python, PyTorch, Scikit-Learn, Pandas, FastAPI
* **Backend:** Java 17, Spring Boot, Spring Web, Hibernate, H2 Database, Docker
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Fetch API
* **Deployment:** Render.com (Backend & AI), GitHub Pages (Frontend)

## Key Features

* **Real-time AI Inference:** The Python microservice generates product recommendations on the fly using a trained Deep Learning model.
* **Fallback Mechanism:** If the AI engine is unreachable or a new user has no purchase history, the Java backend automatically serves a list of generic "Bestseller" products.
* **Automated Data Seeding:** The Java backend automatically fetches the real product catalog from the Python microservice at startup to populate its local database.
* **Jupyter Notebook Documentation:** Includes a comprehensive `notebook.ipynb` notebook detailing the Exploratory Data Analysis (EDA), data cleaning, and PyTorch model training process (achieving a robust Hit Ratio @ 10).

## How to Run Locally

If you wish to run this architecture on your local machine, follow these steps:

### 1. Start the Python AI Engine
Navigate to the `ml-python` directory, install dependencies, and start the FastAPI server:
```bash
cd ml-python
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### 2. Start the Java Backend
Open the backend-java folder in your preferred IDE (IntelliJ IDEA / Eclipse) and run the RecommandationsApplication.java main class. The server will start on port 8080.

### 3. Open the Frontend
Simply open the index.html file located in the frontend folder in any modern web browser.
