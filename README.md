# AI Resume Analyzer & Job Matcher

A polished Streamlit web application that analyzes PDF resumes, extracts key skills, and matches them against a job database using NLP techniques.

## Features
- **PDF Text Extraction**: Uses PyMuPDF for reliable text parsing.
- **Skill Extraction**: Matches resume content against a predefined list of 30+ technical skills.
- **Job Matching**: Combines skill overlap logic with TF-IDF cosine similarity for accurate recommendations.
- **Smart Scoring**: Generates a fitness score (0-100) for various job roles.
- **Missing Skills Identification**: Tells you exactly what to add to your resume to match your top job recommendation.

## Installation

1. **Clone or Download** this project into a folder.
2. **Open a terminal** in that folder.
3. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Run the following command in your terminal:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

## File Structure
- `app.py`: Main application logic and UI.
- `jobs.csv`: Sample database of job roles and requirements.
- `requirements.txt`: List of necessary Python libraries.
- `README.md`: This documentation.

## Skills Tracked
The app currently tracks skills such as: Python, SQL, Machine Learning, React, AWS, Docker, Flask, JavaScript, and many more. To add more, simply update the `PREDEFINED_SKILLS` list in `app.py`.
