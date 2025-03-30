import re
import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import Counter

# Download required NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Sample Descriptions (Updated List)
descriptions = [
    "contract pool id must moved different acount",
    "case transfser credit pool",
    "cardinal health inc able activate ngfw credits need redistribution auth codes",
    "case transfer credit pool csps merge accounts",
    "deactivate vm update credit consumption",
    "remove expired credit pool id",
    "case credit pool discrepancy",
    "case transfer credit pool id csp accounts nswc pcd support",
    "cardinal health credit pool consolidation",
    "license transfer customers ncr voyix ncr aleos",
    "device registration issue"
]

# Step 1: Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [nlp(word)[0].lemma_ for word in tokens]  # Lemmatization
    return ' '.join(tokens)

cleaned_descriptions = [preprocess_text(desc) for desc in descriptions]

# Step 2: Topic Modeling (LDA, LSA, NMF)
def apply_topic_modeling(texts, n_topics=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lsa = TruncatedSVD(n_components=n_topics, random_state=42)
    nmf = NMF(n_components=n_topics, random_state=42)
    
    lda_topics = lda.fit_transform(X)
    lsa_topics = lsa.fit_transform(X)
    nmf_topics = nmf.fit_transform(X)
    
    return {
        "LDA": lda_topics,
        "LSA": lsa_topics,
        "NMF": nmf_topics,
        "Feature_Names": vectorizer.get_feature_names_out()
    }

topic_results = apply_topic_modeling(cleaned_descriptions, n_topics=3)

# Step 3: Generate Semantic Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient BERT model
embeddings = model.encode(cleaned_descriptions)

# Step 4: Clustering Using K-Means
num_clusters = 3  # Adjust based on dataset size
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)
cluster_labels = kmeans.labels_

# Step 5: Assign Descriptive Topic Labels to Clusters
def generate_dynamic_titles(texts, labels, num_keywords=3):
    cluster_keywords = {}
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    for cluster in set(labels):
        cluster_indices = np.where(labels == cluster)
        cluster_texts = X[cluster_indices]
        word_counts = np.array(cluster_texts.sum(axis=0)).flatten()
        top_indices = word_counts.argsort()[-num_keywords:][::-1]
        cluster_keywords[cluster] = [feature_names[i] for i in top_indices]

    cluster_titles = {
        cluster: ' '.join(keywords).title() + " Related Issues"
        for cluster, keywords in cluster_keywords.items()
    }
    
    return cluster_titles

cluster_titles = generate_dynamic_titles(cleaned_descriptions, cluster_labels)

# Step 6: Combine Topic Modeling & Clustering
final_results = pd.DataFrame({
    "Description": descriptions,
    "Cleaned_Text": cleaned_descriptions,
    "Cluster": [cluster_titles[label] for label in cluster_labels]
})

# Display Topics Assigned to Each Cluster
final_clusters = {}
for i, desc in enumerate(descriptions):
    topic_name = cluster_titles[cluster_labels[i]]
    if topic_name not in final_clusters:
        final_clusters[topic_name] = []
    final_clusters[topic_name].append(desc)

# Print Grouped Topics
for topic, texts in final_clusters.items():
    print(f"\nTopic: {topic}")
    for text in texts:
        print(f"  - {text}")
