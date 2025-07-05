import streamlit as st

st.title("HireGenei:Linkedin Jobs Scrapper")

roles=st.session_state["extracted_roles"]
run_scrapper=st.session_state['run']

st.multiselect("Please enter your desired role if not extracted from resume: ",options=st.session_state["selected_roles"],default=[r[0] for r in roles])

cities=st.text_input("Enter cities to scrape from ","Delhi,Mumbai,Pune,Chandigarh")
country=st.text_input("Enter your country","India")
work_types = st.multiselect("Select work types:", options=["On-site", "Hybrid", "Remote"], default=["Remote"])
exp_levels = st.multiselect("Select experience levels:", options=["Intership", "Entry level", "Associate", "Mid-senior level"], default=["Intership"])
time_filter = st.selectbox("Select time filter:", options=["Past 24 hours", "Past week", "Past month"], index=1)

if st.button("Start Scrapping"):
    progress=st.empty()
    table=st.empty()

    for role in roles[0]:
                for status, df in run_scrapper(cities, country, role, work_types, exp_levels, time_filter):
                    progress.write(status)
                    table.dataframe(df)

                st.success("Job Scrapping Completed :)")
                
                 
      