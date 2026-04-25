from flask import Flask, request, render_template, redirect, url_for
import os
import joblib
import re
from pdfminer.high_level import extract_text
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# load tuned pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'placement_pipeline_tuned.joblib')
pipe = joblib.load(MODEL_PATH)

FEATURES = [
    'age','gender','cgpa','branch','college_tier','internships_count','projects_count',
    'certifications_count','coding_skill_score','aptitude_score','communication_skill_score',
    'logical_reasoning_score','hackathons_participated','github_repos','linkedin_connections',
    'mock_interview_score','attendance_percentage','backlogs','extracurricular_score',
    'leadership_score','volunteer_experience','sleep_hours','study_hours_per_day'
]

def extract_features_from_text(text):
    t = text.lower()
    feat = {}
    highlights = {}
    # age
    m = re.search(r'(?:age[:\s]|\bage\s)(\d{2})', t)
    feat['age'] = int(m.group(1)) if m else None
    highlights['age'] = m.group(0) if m else 'Not mentioned'
    # gender
    if 'female' in t and 'male' not in t:
        feat['gender'] = 'Female'
        highlights['gender'] = 'female'
    elif 'male' in t and 'female' not in t:
        feat['gender'] = 'Male'
        highlights['gender'] = 'male'
    else:
        feat['gender'] = 'not mentioned '
        highlights['gender'] = ''
    # cgpa / gpa
    m = re.search(r'(?:cgpa|gpa)[:\s]*([0-9]\.?[0-9]{1,2})', t)
    feat['cgpa'] = float(m.group(1)) if m else 6
    highlights['cgpa'] = m.group(0) if m else ''
    # branch (look for common branches)
    branches = ['cse','it','ece','eee','civil','mechanical']
    found_branch = next((b for b in branches if b in t), None)
    feat['branch'] = found_branch.upper() if found_branch else 'Not mentioned '
    highlights['branch'] = found_branch if found_branch else ''
    # college_tier (fallback Tier 2)
    if 'tier 1' in t:
        feat['college_tier'] = 'Tier 1'
        highlights['college_tier'] = 'tier 1'
    elif 'tier 3' in t:
        feat['college_tier'] = 'Tier 3'
        highlights['college_tier'] = 'tier 3'
    else:
        feat['college_tier'] = 'Not-mentioned'
        highlights['college_tier'] = ''
    # internships/projects/certifications via keyword counts
    interns = t.count('intern')
    feat['internships_count'] = interns
    highlights['internships_count'] = f"{interns} occurrences of 'intern'" if interns else ''
    projects = t.count('project')
    feat['projects_count'] = projects
    highlights['projects_count'] = f"{projects} occurrences of 'project'" if projects else ''
    certs = t.count('certif')
    feat['certifications_count'] = certs
    highlights['certifications_count'] = f"{certs} occurrences of 'certif'" if certs else ''
    # coding skill score: count programming keywords
    langs = ['python','java','c++','c#','javascript','sql','r','matlab','html','css']
    lang_hits = sum(1 for l in langs if l in t)
    feat['coding_skill_score'] = min(100, lang_hits * 15)
    highlights['coding_skill_score'] = ', '.join([l for l in langs if l in t])
    # aptitude/communication/logical/mock interview: heuristics
    a = t.count('aptitude')
    c = t.count('communication')
    r = t.count('reason')
    m_inter = t.count('interview')
    feat['aptitude_score'] = 60 + min(20, a*5)
    feat['communication_skill_score'] = 60 + min(20, c*5)
    feat['logical_reasoning_score'] = 60 + min(20, r*5)
    feat['mock_interview_score'] = 60 + min(20, m_inter*5)
    highlights['aptitude_score'] = "aptitude" if a else ''
    highlights['communication_skill_score'] = "communication" if c else ''
    highlights['logical_reasoning_score'] = "reason" if r else ''
    highlights['mock_interview_score'] = "interview" if m_inter else ''
    # hackathons, github, linkedin
    hack = t.count('hackathon')
    feat['hackathons_participated'] = hack
    highlights['hackathons_participated'] = f"{hack} occurrences" if hack else ''
    gh = 'github.com' in t
    feat['github_repos'] = 1 if gh else 0
    highlights['github_repos'] = 'github.com' if gh else ''
    li = 'linkedin.com' in t
    feat['linkedin_connections'] = 1 if li else 0
    highlights['linkedin_connections'] = 'linkedin.com' if li else ''
    # attendance/backlogs: default
    feat['attendance_percentage'] = 75.0
    highlights['attendance_percentage'] = ''
    # attempt to find backlogs
    m_back = re.search(r'backlog[s]?[:\s]?(\d+)', t)
    feat['backlogs'] = int(m_back.group(1)) if m_back else 0
    highlights['backlogs'] = m_back.group(0) if m_back else ''
    #extracurricular & leadership & volunteer
    ex = min(100, t.count('extra')*20 + t.count('club')*10)
    feat['extracurricular_score'] = ex
    highlights['extracurricular_score'] = 'extra/club' if ex else ''
    lead = any(k in t for k in ['captain','president','lead','head'])
    feat['leadership_score'] = 20 if lead else 0
    highlights['leadership_score'] = 'leadership term' if lead else ''
    vol = 'volunteer' in t
    feat['volunteer_experience'] = 'Yes' if vol else 'No'
    highlights['volunteer_experience'] = 'volunteer' if vol else ''
    # sleep & study hours: defaults
    feat['sleep_hours'] = 7.0
    highlights['sleep_hours'] = ''
    feat['study_hours_per_day'] = 3.0
    highlights['study_hours_per_day'] = ''
    # certifications/hackathon numeric fallbacks ensured
    for k in FEATURES:
        if k not in feat:
            feat[k] = 0
        if k not in highlights:
            highlights[k] = ''
    return feat, highlights

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # If accessed by GET (e.g., user navigated directly), send back to upload page
    if request.method == 'GET':
        return redirect(url_for('index'))

    if 'resume' not in request.files:
        return redirect(url_for('index'))
    f = request.files['resume']
    if f.filename == '':
        return redirect(url_for('index'))
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(save_path)
    text = extract_text(save_path)
    feats, highlights = extract_features_from_text(text)
    X = pd.DataFrame([feats])
    # enforce minimum CGPA criteria before prediction
    try:
        cgpa_val = float(feats.get('cgpa', 0))
    except Exception:
        cgpa_val = 0.0
    if cgpa_val < 6.0:
        # Candidate does not meet minimum CGPA requirement; skip model
        return render_template('result.html', prediction='Not Eligible (CGPA < 6.0)', probability=None, features=feats, highlights=highlights)

    # ensure column order expected by pipeline
    # pipeline preprocessor will handle types
    pred = pipe.predict(X)[0]
    try:
        prob = pipe.predict_proba(X)[0][1]
    except Exception:
        prob = None
    return render_template('result.html', prediction=('Eligible for P lacement ' if pred==1 else 'Not Eligible for Placement'), probability=prob, features=feats, highlights=highlights)

if __name__ == '__main__':
    app.run(debug=True)
