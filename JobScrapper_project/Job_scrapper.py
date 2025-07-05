import requests
from bs4 import BeautifulSoup
import pandas as pd

from flair.models import SequenceTagger  # flair used to get skill detection
from flair.data import Sentence

import streamlit as st

import threading
import time
import random
import os
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load flair's skill detection model
flair_model = SequenceTagger.load("kaliani/flair-ner-skill")

# Map experience levels to LinkedIn URL codes
experience_level_mapping = {
    "Intership": "f_E=1",
    "Entry level": "f_E=2",
    "Associate": "f_E=3",
    "Mid-senior level": "f_E=4"
}

# Map work type to URL codes
work_type_mapping = {
    "On-site": "f_WT=1",
    "Hybrid": "f_WT=2",
    "Remote": "f_WT=3",
}

time_filter_mapping = {
    "Past 24 hours": " f_TPR=r86400",
    "Past week": " f_TPR=r604800",
    "Past month": " f_TPR=r2592000",
}

# Define function to find skills in description of jobs
def get_skills(text):
    sentence = Sentence(text)
    flair_model.predict(sentence)
    return [entity.text for entity in sentence.get_spans("ner")]

# Example to scrape skills (can be removed in production)
description = """Quantum IT Innovation is a globally recognized mobile & web app development and digital marketing agency. It is a USA-based incorporation with offices in the US and the UK and a delivery center based in India with a team of more than 100 members. We provide complete web development and digital

Selected Intern's Day-to-day Responsibilities Include

 Collaborate with our team of experts to research and develop innovative algorithms for quantum computing applications
 Utilize your expertise in Generative AI to design and implement machine learning models for quantum information processing
 Assist in the optimization and training of AI algorithms to improve performance and efficiency
 Contribute to the analysis and interpretation of data to drive actionable insights and innovations
 Stay current with the latest advancements in AI and quantum computing to enhance project outcomes
 Engage in hands-on experimentation and testing to validate the effectiveness of AI solutions
 Communicate your findings and recommendations effectively to contribute to the overall success of our projects

Skill(s) required

AI Image Generation Artificial intelligence Data Analysis Machine Learning

Other Requirements

 Candidates must have done at least one internship in the same field
 Candidates must be 2023-2025 (with no regular classes) are only preferred
 Skills required: Artificial intelligence and machine learning

Note: This is a paid internship.Skills: data analysis,ai image generation,data,algorithms,it,intelligence,machine learning,artificial intelligence"""
# get_skills(description)  # Example usage

# ScraperManager class
class ScraperManager:
    def __init__(self):
        self.stop_event = threading.Event()
        self.current_df = pd.DataFrame()
        self.lock = threading.Lock()

    def reset(self):
        self.stop_event.clear()
        self.current_df = pd.DataFrame()

    def add_job(self, job_data):
        with self.lock:
            new_df = pd.DataFrame([job_data])
            self.current_df = pd.concat([self.current_df, new_df], ignore_index=True)

scraper_manager = ScraperManager()

# Define function to save jobs
def save_csv(df, filename="jobs"):
    try:
        os.makedirs("saved_jobs", exist_ok=True)
        if not filename:
            filename = f"jobs_{int(time.time())}"
        full_path = f"saved_jobs/{filename}.csv"
        df.to_csv(full_path, index=False)
        return f"Saved to {full_path}"
    except Exception as e:
        return f"Save error: {str(e)}"

# Process Job function
def process_job(job, work_type, exp_level, position):
    try:
        title_element = job.find('h3', class_='base-search-card__title')
        company_element = job.find('a', class_='hidden-nested-link')
        loc_element = job.find('span', class_='job-search-card__location')
        link_element = job.find('a', class_='base-card__full-link')

        if not all([title_element, company_element, loc_element, link_element]):
            return None

        title = title_element.text.strip()
        company = company_element.text.strip()
        location = loc_element.text.strip()
        link = link_element['href'].split('?')[0]

        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        desc = "description not available"
        skill = []

        try:
            time.sleep(random.uniform(2, 5))
            response = session.get(
                link,
                headers={
                    'User-Agent': random.choice([
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4515.107 Safari/537.36'
                    ]),
                    'Accept-Language': 'en-US,en;q=0.9',
                },
                timeout=10
            )
            job_soup = BeautifulSoup(response.text, 'html.parser')
            description_selectors = [
                'div.description__text',
                'div.show-more-less-html__markup',
                'div.core-section-container__content',
                'section.core-section-container',
            ]
            for selector in description_selectors:
                desc_element = job_soup.select_one(selector)
                if desc_element:
                    desc = desc_element.get_text('\n').strip()
                    skill = get_skills(desc)
                    break
        except Exception as e:
            print(f"Error fetching job page for {link}: {str(e)}")

    except Exception as e:
        print(f"Error processing job card: {str(e)}")

    return {
        "Position": position,
        "Date": datetime.now().strftime('%Y-%m-%d'),
        "Work type": work_type,
        "Level": exp_level,
        "Title": title,
        "Company": company,
        "Location": location,
        "Link": f"[{link}]({link})",
        "Description": desc,
        "Skills": ", ".join(skill[:5]) if skill else "No skills detected"
    }

# Scrap Job function
def scrape_jobs(location, position, work_types, exp_levels, time_filter):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for work_type in work_types:
        for exp_level in exp_levels:
            if scraper_manager.stop_event.is_set():
                return

            try:
                base_url = (
                    f"https://www.linkedin.com/jobs/search/?keywords={position}&location={location}"
                    f"&{work_type_mapping[work_type]}"
                    f"&{experience_level_mapping[exp_level]}"
                    f"&{time_filter_mapping[time_filter]}"
                    f"&radius=0"
                )
                try:
                    response = session.get(base_url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    count_text = soup.find('span', class_='results-context-header__job-count')
                    total_jobs = int(count_text.text.replace(',', '').replace('.', '').strip()) if count_text else 25
                except Exception:
                    total_jobs = 25

                total_jobs = min(total_jobs, 100)

                for start in range(0, total_jobs, 25):
                    if scraper_manager.stop_event.is_set():
                        return

                    time.sleep(random.uniform(2, 5))
                    url = f"{base_url}&start={start}"

                    try:
                        response = session.get(url, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        jobs = soup.find_all('div', class_='base-card')
                    except Exception as e:
                        print(f"Failed to scrape page {start}: {str(e)}")
                        continue

                    random.shuffle(jobs)

                    for job in jobs:
                        if scraper_manager.stop_event.is_set():
                            return

                        job_data = process_job(job, work_type, exp_level, position)
                        if job_data:
                            scraper_manager.add_job(job_data)
                            yield
            except Exception as e:
                print(f"Scraping error: {str(e)}")

# Run scrapper function
def run_scrapper(cities, states, positions, work_types, exp_levels, time_filter):
    scraper_manager.reset()
    cities_list = [c.strip() for c in cities.split(',') if c.strip()]
    states_list = [s.strip() for s in states.split(',') if s.strip()]
    location = [f"{city},{state}" for city in cities_list for state in states_list]
    positions_list = [str(p).strip().replace(' ', '%20') for p in positions if isinstance(p, str) and p.strip()]



    def worker():
        for loc in location:
            for pos in positions_list:
                if scraper_manager.stop_event.is_set():
                    return
                for _ in scrape_jobs(loc, pos, work_types, exp_levels, time_filter):
                    pass

    thread = threading.Thread(target=worker)
    thread.start()

    while thread.is_alive():
        time.sleep(0.5)
        with scraper_manager.lock:
            yield 'Scrapping in progress...', scraper_manager.current_df
    yield "Scrapping Completed!" if not scraper_manager.stop_event.is_set() else "Scrapping stopped!", scraper_manager.current_df


       