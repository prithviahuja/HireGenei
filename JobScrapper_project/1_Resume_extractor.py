import streamlit as st
st.set_page_config(page_title="HireGenei", layout="wide")

import Roles_extractor
import Skills_Extractor
from sentence_transformers import SentenceTransformer, util
import os
import tempfile
from Job_scrapper import run_scrapper, scraper_manager, save_csv  
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

st.title("HireGenei:Resume to Job Scrapper")

uploaded_file=st.file_uploader("Drop your pdf here",type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path=tmp.name

        st.success("File uploaded Succesfully")    
        st.session_state["loader"]= PyPDFLoader(tmp_path)
        skills=list(Skills_Extractor.skills_extraction(tmp_path))
        st.write("### Skills Extracted: ",skills)
        st.session_state["extracted_skills"]=skills

        roles=Roles_extractor.roles_score(skills)
        st.write("### Matching Roles: ") ### Makes it a heading
        i=1

        for role in roles:
            st.write(f"{i}.***{role[0]}***") # *** makes the text bold
            i=i+1

        st.session_state["extracted_roles"]= roles 
        st.session_state["selected_roles"]= list(Roles_extractor.job_roles.keys())    

st.session_state['run']=run_scrapper
