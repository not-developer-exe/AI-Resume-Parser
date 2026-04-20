import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Our tiny training dataset
data = [
    ("MERN stack developer, React, Node.js, MongoDB, Express, API", "Software Engineering"),
    ("Python, Machine Learning, Data Analysis, SQL, TensorFlow", "Data Science"),
    ("SEO, digital marketing, content strategy, social media", "Marketing"),
    ("Figma, UI/UX design, wireframing, Adobe XD, prototyping", "Design"),
    ("Financial modeling, accounting, Excel, risk analysis", "Finance")
]
X_train, y_train = zip(*data)

# 2. Build the ML Pipeline (Converts words to math, then trains the AI)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# 3. Save the actual model file to your folder
with open("ml_job_classifier.pkl", "wb") as file:
    pickle.dump(model, file)

print("ML Model 'ml_job_classifier.pkl' created successfully!")