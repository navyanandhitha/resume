import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Predefined keyword sets
expected_skills = ['python', 'machine learning', 'data analysis', 'communication', 'sql']
expected_experience = ['internship', 'project', 'work experience', 'training']
expected_education = ['bachelor', 'degree', 'university', 'college', 'graduation']

# Function to extract text
def extract_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_file:
        for page in pdf_file.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.lower()
    return text

# Count keyword matches
def score_section(text, keywords):
    return sum(1 for kw in keywords if kw in text)

# Streamlit app
st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")
st.title("Smart Resume Analyzer")
st.write("Upload your resume and paste the job description to see if you're a match.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description Here")

if resume_file and job_desc:
    resume_text = extract_text(resume_file)
    job_desc = job_desc.lower()

    # Score calculation
    skill_score = score_section(resume_text, expected_skills)
    exp_score = score_section(resume_text, expected_experience)
    edu_score = score_section(resume_text, expected_education)
    similarity_score = cosine_similarity(
        CountVectorizer(stop_words='english').fit_transform([resume_text, job_desc])
    )[0][1]

    total = 3 + 1  # 3 sections + similarity
    overall_score = round(((skill_score / len(expected_skills)) +
                           (exp_score / len(expected_experience)) +
                           (edu_score / len(expected_education)) +
                           similarity_score) / total * 100, 2)

    st.subheader("Analysis Report")
    st.write(f"*Skill Match:* {skill_score}/{len(expected_skills)}")
    st.write(f"*Experience Match:* {exp_score}/{len(expected_experience)}")
    st.write(f"*Education Match:* {edu_score}/{len(expected_education)}")
    st.write(f"*Relevance to Job Description:* {round(similarity_score * 100, 2)}%")

    st.subheader("Final Evaluation")
    st.progress(overall_score/100)
    st.success(f"*Overall Score: {overall_score}%*")

    threshold = 70  # Eligibility cutoff
    if overall_score >= threshold:
        st.balloons()
        st.markdown("### ✅ Congratulations! You're eligible for this job.")
    else:
        st.warning("⚠ You're not eligible yet. Here are some suggestions:")
        if skill_score < len(expected_skills):
            missing = [s for s in expected_skills if s not in resume_text]
            st.info(f"*Add these missing skills:* {', '.join(missing)}")
        if exp_score < len(expected_experience):
            st.info("*Include more project/internship/work experience details.*")
        if edu_score < len(expected_education):
            st.info("*Mention your degree, university, or graduation info clearly.*")
        if similarity_score < 0.5:
            st.info("*Improve keyword match with the job description. Use relevant terms.*")
    
