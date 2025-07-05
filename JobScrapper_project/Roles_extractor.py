from sentence_transformers import SentenceTransformer,util
import Skills_Extractor

model = SentenceTransformer('all-MiniLM-L6-v2')

job_roles = {
    # Core AI & Data Roles
    "Data Scientist": [
        "python", "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scikitlearn", "xgboost", "lightgbm",
        "catboost", "mlflow", "optuna", "statistics", "dask", "vaex", "joblib"
    ],
    "Data Analyst": [
        "sql", "excel", "powerbi", "tableau", "looker", "superset", "pandas", "numpy", "matplotlib", "seaborn",
        "statistics"
    ],
    "ML Engineer": [
        "python", "tensorflow", "keras", "pytorch", "mlflow", "onnx", "huggingface", "transformers", "joblib",
        "optuna", "torchserve", "tfserving", "gradio", "streamlit", "sagemaker", "vertexai"
    ],
    "Data Engineer": [
        "spark", "hadoop", "hive", "kafka", "apachebeam", "airflow", "deltalake", "snowflake", "glue",
        "presto", "flink", "mysql", "postgresql", "mongodb", "neo4j", "cassandra", "dynamodb", "influxdb",
        "supabase"
    ],

    # Generative AI Roles
    "Generative AI Engineer": [
        "transformers", "huggingface", "gpt", "bert", "t5", "lora", "qlora", "peft", "langchain",
        "openaiai", "crewai", "langraph", "lanserve", "rag", "weaviate", "pinecone", "haystack", "vertexai"
    ],
    "NLP Engineer": [
        "spacy", "nltk", "gensim", "huggingface", "transformers", "bert", "t5", "marianmt", "fairseq", "gpt",
        "langchain", "rag", "peft", "haystack", "openaiai"
    ],
    "Computer Vision Engineer": [
        "opencv", "yolo", "detectron2", "mediapipe", "paddleocr", "tesseract", "cnn"
    ],

    # Full Stack / Web Dev Roles
    "Web Developer": [
        "html", "css", "javascript", "react", "vuejs", "angular", "svelte", "nextjs", "tailwind", "bootstrap",
        "nodejs", "expressjs", "restapi", "graphql", "grpc", "websockets", "flask", "django", "fastapi"
    ],
    "Backend Developer": [
        "nodejs", "expressjs", "flask", "django", "fastapi", "springboot", "dotnet", "graphql", "grpc",
        "restapi", "mysql", "postgresql", "mongodb", "redis", "neo4j"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "typescript", "react", "angular", "vuejs", "svelte", "nextjs", "tailwind",
        "bootstrap"
    ],
    "Mobile App Developer": [
        "flutter", "reactnative", "android", "ios", "dart", "kotlin", "swift"
    ],

    # DevOps & Cloud Roles
    "DevOps Engineer": [
        "git", "github", "gitlab", "bitbucket", "jenkins", "docker", "kubernetes", "helm", "istio",
        "ansible", "terraform", "azuredevops", "argocd", "prometheus", "grafana"
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "s3", "ec2", "lambda", "cloudfunctions", "firebase", "cloudformation",
        "cloudflare"
    ],
    "MLOps Engineer": [
        "mlflow", "bentoml", "torchserve", "sagemaker", "vertexai", "onnx", "tfserving", "gradio", "streamlit",
        "docker", "kubernetes"
    ],

    # Specialized Roles
    "AI Researcher": [
        "pytorch", "tensorflow", "huggingface", "gpt", "bert", "t5", "cnn", "rnn", "lstm", "transformers",
        "peft", "qlora", "fairseq", "marianmt"
    ],
    "Recommender Systems Engineer": [
        "surprise", "lightfm", "implicit", "faiss", "annoy", "milvus"
    ],
    "Cybersecurity Engineer": [
        "nmap", "wireshark", "metasploit", "burpsuite", "owasp", "jwt", "oauth2", "saml", "ssltls"
    ],
    "Game Developer": [
        "unity", "unrealengine", "godot", "blender", "threejs"
    ],
    "Blockchain Developer": [
        "solidity", "web3js", "ethersjs", "truffle", "hardhat", "ganache", "ipfs"
    ],
    "Automation Specialist / No-Code Developer": [
        "zapier", "integromat", "powerapps", "bubble", "ifttt"
    ],
    "System Architect": [
        "loadbalancing", "messagequeues", "caching", "apigateway", "microservices", "monolith",
        "eventdriven", "pubsub", "modulefederation"
    ],
    "Payment Integration Engineer": [
        "stripe", "razorpay", "paypal"
    ]
}

#Create a list of all the roles form the disctionary
roles=list(job_roles.keys())
roles

def roles_score(user_input):
    user_text=" ".join(user_input) #creates a sentence showing all the inputs we are getting from user
    user_text

    roles_text=[" ".join(value) for value in job_roles.values()]
    roles_text #makes a list of values from all roles 

    user_embed = model.encode(user_text, convert_to_tensor=True)
    role_embeds = model.encode(roles_text, convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embed, role_embeds)[0] #we used 0 to make it 1d

    scores=cosine_scores.tolist() #convert tensor scores to list

    roles_scores=[]
    for i in range(len(roles)):
        roles_scores.append((roles[i],scores[i]))# create list having roles and scores

    roles_scores.sort(key=lambda x: x[1], reverse=True) 

    return roles_scores[:5]

user_input=["skitlearn","langchain","pandas","LLms","tensorflow","docker", "kubernetes", "aws"]
#user_input=final_func.skills_extraction()
roles_score(user_input)    