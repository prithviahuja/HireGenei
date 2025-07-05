import fitz
import spacy
import re
import pdfplumber
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util

# Step 1: Define curated skill list (can expand later)
skills = [
    # Programming Languages
    "python", "cpp", "csharp", "java", "javascript", "typescript", "dart", "go", "rust", "r", "shell", "bash", "sql",
    "scala", "php", "perl", "matlab", "assembly", "swift", "kotlin",

    # Web Development
    "html", "css", "react", "angular", "vuejs", "svelte", "nextjs", "tailwind", "bootstrap",
    "nodejs", "expressjs", "django", "flask", "fastapi", "springboot", "dotnet", "laravel",
    "graphql", "grpc", "websockets", "restapi",

    # Mobile Development
    "flutter", "reactnative", "android", "ios",

    # Databases
    "mysql", "postgresql", "sqlite", "mongodb", "redis", "neo4j", "cassandra", "dynamodb",
    "influxdb", "firestore", "supabase",

    # Data Engineering & Big Data
    "hadoop", "hive", "spark", "kafka", "apachebeam", "airflow", "deltalake", "snowflake",
    "glue", "presto", "flink",

    # DevOps & CI/CD
    "git", "github", "gitlab", "bitbucket", "jenkins", "docker", "kubernetes", "helm", "istio",
    "ansible", "terraform", "azuredevops", "argocd", "prometheus", "grafana",

    # Cloud Platforms
    "aws", "azure", "gcp", "s3", "ec2", "lambda", "cloudfunctions", "firebase", "cloudflare", "cloudformation",

    # Data Analysis & Visualization
    "numpy", "pandas", "matplotlib", "seaborn", "plotly", "dask", "vaex",
    "powerbi", "tableau", "looker", "superset",

    # Machine Learning
    "scikitlearn", "xgboost", "lightgbm", "catboost", "mlflow", "optuna", "joblib", "ann", "cnn", "rnn", "lstm",

    # Deep Learning
    "tensorflow", "keras", "pytorch", "huggingface", "transformers", "t5", "bert", "gpt", "peft", "lora", "qlora",

    # Natural Language Processing (NLP)
    "spacy", "nltk", "gensim", "langchain", "openaiai", "fairseq", "marianmt", "crewai", "haystack",

    # Computer Vision
    "opencv", "yolo", "detectron2", "mediapipe", "paddleocr", "tesseract",

    # MLOps & Model Deployment
    "bentoml", "torchserve", "sagemaker", "vertexai", "onnx", "tfserving", "gradio", "streamlit","langraph","lanserve"

    # Recommendation Systems
    "surprise", "lightfm", "implicit", "faiss", "annoy", "milvus",

    # System Design & Architecture
    "loadbalancing", "messagequeues", "caching", "apigateway", "microservices", "monolith",
    "eventdriven", "pubsub", "modulefederation",

    # Payment Integration
    "stripe", "razorpay", "paypal",

    # Automation & No-Code Tools
    "zapier", "integromat", "powerapps", "bubble", "ifttt",

    # Web3 & Blockchain
    "solidity", "web3js", "ethersjs", "truffle", "hardhat", "ganache", "ipfs",

    # Game Development & Graphics
    "unity", "unrealengine", "godot", "blender", "threejs",

    # Cybersecurity
    "nmap", "wireshark", "metasploit", "burpsuite", "owasp", "jwt", "oauth2", "saml", "ssltls",

    # AI Agents & Hybrid Search Systems
    "aiagents", "rag", "weaviate", "pinecone", "pyg", "stablebaselines3", "cypher"
]


# Preprocess text
def clean_resume_text(text):
    text = text.replace('\u200b', ' ')
    text = re.sub(r'\n+', ' ', text)               # Remove line breaks
    text = re.sub(r'●', ' ', text)                 # Replace bullets
    text = re.sub(r'\s+', ' ', text)               # Collapse whitespace
    return text.lower().strip()

# Preprocess skills
def clean_term(term):
    return re.sub(r"[^a-zA-Z0-9\#\+\-\.]", " ", term.lower()).strip()

# Step 2: Extract text using pdfplumber
pdf_path = r"C:\Users\Prithvi Ahuja\Downloads\Vanshaj_Resume_15052025.pdf"
def skills_extraction(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    raw_text = clean_resume_text(raw_text)

    # Step 3: Exact skill match (multi-word safe)
    exact_matches = set()
    for skill in skills:
        pattern = re.escape(skill.lower())
        if re.search(r'\b' + pattern + r'\b', raw_text):
            exact_matches.add(skill)

    # Step 4: Fuzzy matching
    words = set(re.findall(r"\b[a-zA-Z\#\+\-\.]{2,}\b", raw_text))
    fuzzy_matches = set()
    for word in words:
        match = process.extractOne(word, skills, score_cutoff=93)  # stricter
        if match:
            fuzzy_matches.add(match[0])

    # Step 5: Semantic matching
    model = SentenceTransformer('all-MiniLM-L6-v2')
    skill_embeddings = model.encode(skills, convert_to_tensor=True)
    text_embeddings = model.encode(list(words), convert_to_tensor=True)

    semantic_matches = set()
    threshold = 0.75
    for i, vec in enumerate(text_embeddings):
        scores = util.cos_sim(vec, skill_embeddings)[0]
        best_idx = scores.argmax().item()
        if float(scores[best_idx]) >= threshold:
            semantic_matches.add(skills[best_idx])

    # Step 6: Combine all
    final_skills = exact_matches.union(fuzzy_matches).union(semantic_matches)

    return final_skills
