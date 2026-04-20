import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Resume Analyzer & Job Matcher",
    page_icon="📄",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    /* Global Styles */
    .main {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Card Design */
    .job-card {
        padding: 24px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .job-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    
    /* Force specific colors for legibility in all themes */
    .job-title {
        color: #1e293b !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-bottom: 8px !important;
    }
    
    .job-skills {
        color: #475569 !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }
    
    .skill-tag {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        background-color: #eff6ff;
        border: 1px solid #dbeafe;
        color: #1e40af !important;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 500;
    }
    
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CATEGORIZED SKILLS DATABASE ---
SKILL_CATEGORIES = {
    "Technology": ["Python", "JavaScript", "Java", "C++", "C#", "SQL", "TypeScript", "Go", "Rust", "Swift", "Kotlin", "PHP", "Ruby", "HTML", "CSS", "Bash", "R", "Dart", "Scala", "Docker", "Kubernetes", "Terraform", "Jenkins", "Ansible", "Git", "CI/CD", "Linux", "Nginx", "Firebase", "Heroku", "Cloudflare", "React", "Angular", "Vue", "Next.js", "Django", "Flask", "FastAPI", "Spring Boot", "Express.js", "Node.js", "Svelte", "Tailwind CSS", "Bootstrap", "jQuery", "Laravel", "Ruby on Rails"],
    "Data Science": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "Matplotlib", "Seaborn", "OpenCV", "Transformers", "LLM", "GNN", "Power BI", "Tableau", "Excel", "Spark", "Hadoop", "Data Analysis", "ETL", "Data Warehouse"],
    "Architecture": ["AutoCAD", "Revit", "SketchUp", "BIM", "3ds Max", "V-Ray", "Rhino", "Grasshopper", "Lumion", "Architecture Design", "Urban Planning", "Interior Design", "Photoshop"],
    "Civil Engineering": ["Construction", "Structural Analysis", "SAP2000", "ETABS", "Concrete Design", "Steel Design", "Project Management", "Site Engineering", "Surveying", "Geotechnical Engineering", "Transportation Engineering"],
    "Business & Management": ["Business Analysis", "Agile", "Scrum", "Marketing", "Finance", "Management", "Strategy", "CRM", "Jira", "Trello", "Agile", "Problem Solving", "Leadership", "Communication", "Requirements Gathering"]
}

# Flatten for extraction
PREDEFINED_SKILLS = [skill for sublist in SKILL_CATEGORIES.values() for skill in sublist]

# Domain mapping for detection
DOMAIN_MAP = {
    "Technology": "Tech",
    "Data Science": "Data",
    "Architecture": "Architecture",
    "Civil Engineering": "Civil",
    "Business & Management": "Business"
}

# Helper for Smart Synonym Handling
SYNONYMS = {
    "react": r"react\.js|reactjs",
    "node.js": r"nodejs|node\.js|node js",
    "next.js": r"nextjs|next\.js",
    "vue": r"vuejs|vue\.js",
    "laravel": r"php laravel",
    "postgres": r"postgresql|postgres",
    "mongo": r"mongodb|mongo"
}

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file using PyMuPDF."""
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def clean_text(text):
    """Cleans extracted text by removing unwanted characters and normalizing whitespace."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespaces with single space
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text.strip().lower()

def extract_skills(text):
    """Extracts skills with smart synonym and boundary handling."""
    found_skills = []
    text_lower = text.lower()
    
    for skill in PREDEFINED_SKILLS:
        skill_lower = skill.lower()
        # Check for synonyms first
        if skill_lower in SYNONYMS:
            pattern = r'\b(' + SYNONYMS[skill_lower] + r')\b'
        else:
            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
        if re.search(pattern, text_lower):
            found_skills.append(skill)
            
    return sorted(list(set(found_skills)))

def detect_domain(extracted_skills):
    """Analyzes extracted skills to determine the dominant industry domain."""
    domain_counts = {category: 0 for category in SKILL_CATEGORIES.keys()}
    
    for skill in extracted_skills:
        for category, skills_list in SKILL_CATEGORIES.items():
            if skill in skills_list:
                domain_counts[category] += 1
                
    # Select the category with the most matches
    dominant_category = max(domain_counts, key=domain_counts.get)
    
    # If no skills found, default to Business/General
    if domain_counts[dominant_category] == 0:
        return "Business"
        
    return DOMAIN_MAP.get(dominant_category, "Business")

def match_jobs(domain, extracted_skills, jobs_df):
    """Targeted matching: ONLY matches jobs within the detected domain using overlap formula."""
    try:
        # Filter jobs by detected domain
        relevant_jobs = jobs_df[jobs_df['Domain'] == domain]
        
        if relevant_jobs.empty:
            # Fallback if no specific jobs in domain (return all as general match)
            relevant_jobs = jobs_df
            
        results = []
        user_skills_set = set([s.lower() for s in extracted_skills])
        
        for _, row in relevant_jobs.iterrows():
            job_skills = [s.strip().lower() for s in row['Required Skills'].split(',')]
            job_skills_set = set(job_skills)
            
            # Formula: (matched_skills / total_job_skills) * 100
            matches = user_skills_set.intersection(job_skills_set)
            score = (len(matches) / len(job_skills_set)) * 100 if job_skills_set else 0
            
            results.append({
                "title": row['Job Title'],
                "skills": row['Required Skills'],
                "score": round(score, 1)
            })
            
        # Sort and return top 5
        return sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    except Exception as e:
        st.error(f"Error in job matching: {e}")
        return []

# --- MAIN APP ---

def main():
    st.title("🚀 AI Resume Analyzer & Job Matcher")
    st.subheader("Smart talent matching for the modern workforce.")
    
    st.divider()
    
    # --- SESSION STATE INITIALIZATION ---
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    def reset_app():
        st.session_state.uploader_key += 1
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("Upload your resume in PDF format to see how well you match with industry-standard job roles.")
        
        if st.button("🔄 Reset Context", help="Clear current upload and reset all results"):
            reset_app()
            
        st.markdown("---")
        st.markdown("**Version:** 1.1.0")
        st.markdown("**Author:** Senior Python Developer")

    # Load Jobs Data
    try:
        jobs_df = pd.read_csv("jobs.csv")
    except FileNotFoundError:
        st.error("Jobs database (jobs.csv) not found. Please ensure it exists.")
        return

    # Upload Section
    uploaded_file = st.file_uploader(
        "Upload your Resume (PDF)", 
        type="pdf", 
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        with st.spinner("Analyzing your resume..."):
            # Extraction
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if raw_text:
                cleaned_text = clean_text(raw_text)
                extracted_skills = extract_skills(raw_text)
                
                # Layout for results
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.header("📋 Analysis Results")
                    
                    # --- DOMAIN DISPLAY ---
                    detected_domain = detect_domain(extracted_skills)
                    st.markdown(f"""
                        <div style="background-color: #3b82f6; color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold; margin-bottom: 20px;">
                            Industry Domain: {detected_domain}
                        </div>
                    """, unsafe_allow_html=True)

                    st.subheader("Extracted Skills")
                    if extracted_skills:
                        skill_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in extracted_skills])
                        st.markdown(skill_html, unsafe_allow_html=True)
                    else:
                        st.warning("No skills detected from the predefined list.")

                    st.subheader("Resume Summary")
                    st.write(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)

                with col2:
                    st.header("🎯 Target Recommendations")
                    
                    # Targeted Domain-Based Matching
                    recommendations = match_jobs(detected_domain, extracted_skills, jobs_df)
                    
                    for rec in recommendations:
                        with st.container():
                            st.markdown(f"""
                            <div class="job-card">
                                <div class="job-title">🎯 {rec['title']}</div>
                                <div class="job-skills"><strong>Requirements:</strong> {rec['skills']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Use columns for score display to make it cleaner
                            sc1, sc2 = st.columns([4, 1])
                            with sc1:
                                st.progress(rec['score'] / 100)
                            with sc2:
                                st.markdown(f"<p style='color: #1e40af; font-weight: bold; margin-top: -5px;'>{rec['score']}%</p>", unsafe_allow_html=True)

                # Overall Resume Quality Score
                st.divider()
                st.header("📊 Overall Resume Fitness")
                
                avg_top_score = sum([r['score'] for r in recommendations]) / len(recommendations) if recommendations else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Skills Found", len(extracted_skills))
                m2.metric("Best Match", f"{recommendations[0]['score']}%" if recommendations else "0%")
                m3.metric("Overall Score", f"{round(avg_top_score, 1)}/100")

                # Missing Skills logic for the top recommendation
                if recommendations:
                    best_job = recommendations[0]
                    best_job_skills = set([s.strip().lower() for s in best_job['skills'].split(',')])
                    user_skills = set([s.lower() for s in extracted_skills])
                    missing = best_job_skills - user_skills
                    
                    if missing:
                        st.subheader(f"💡 Tips to improve for '{best_job['title']}'")
                        st.write("Consider adding these missing skills to your resume:")
                        missing_html = "".join([f'<span class="skill-tag" style="background-color: #ffe3e3;">{s.title()}</span>' for s in missing])
                        st.markdown(missing_html, unsafe_allow_html=True)
                    else:
                        st.success("You have all the required skills for your top recommendation!")

            else:
                st.error("Could not extract text from the PDF. Please try a different file.")
    else:
        st.info("Welcome! Please upload a PDF resume to get started.")

if __name__ == "__main__":
    main()
