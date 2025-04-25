import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Cache stopwords for performance
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load dataset
def load_data():
    data = pd.read_csv("Amazon_Reviews.csv")
    if 'Review' not in data.columns or 'Sentiment' not in data.columns:
        st.error("Dataset must contain 'Review' and 'Sentiment' columns!")
        return None
    data.dropna(subset=['Review', 'Sentiment'], inplace=True)
    return data

# Train model
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Sentiment'], test_size=0.2, random_state=42)
    model = Pipeline([
        ('preprocess', FunctionTransformer(lambda x: [preprocess_text(text) for text in x], validate=False)),
        ('vectorizer', CountVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return model, acc, y_test, predictions

# Predict sentiment
def predict_sentiment(model, review):
    return model.predict([review])[0]  # raw text passed, preprocessing handled by pipeline

# Word cloud
def generate_wordcloud(data, theme="Light"):
    text = " ".join([preprocess_text(review) for review in data['Review'].dropna()])
    bg_color = 'white' if theme == "Light" else 'black'
    wordcloud = WordCloud(stopwords=stop_words, background_color=bg_color, max_words=100).generate(text)
    return wordcloud

# Analyze uploaded reviews
def analyze_uploaded_file(file, model):
    new_data = pd.read_csv(file)
    if 'Review' not in new_data.columns:
        st.error("Uploaded CSV must contain a 'Review' column.")
        return None
    new_data.dropna(subset=['Review'], inplace=True)
    new_data['Predicted_Sentiment'] = model.predict(new_data['Review'])
    return new_data

# Main App
def main():
    st.set_page_config(page_title="Sentiment Analysis", layout="wide")

    # Theme toggle
    theme = st.radio("🌗 Select Theme:", ["Light", "Dark"], horizontal=True)

    # Apply theme styles
    if theme == "Dark":
        st.markdown("""
            <style>
                body { background-color: #121212; color: #ffffff; }
                .stApp { background-color: #121212; color: #ffffff; }
                .st-cb, .st-emotion-cache-16txtl3, .st-emotion-cache-1r6slb0 {
                    color: white !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                body { background-color: #ffffff; color: #000000; }
                .stApp { background-color: #ffffff; color: #000000; }
            </style>
        """, unsafe_allow_html=True)

    st.title("🔍 Enhanced Customer Review Sentiment Analysis")

    # Load and train
    with st.spinner("Loading and training model..."):
        data = load_data()
        if data is None:
            return
        model, acc, y_test, predictions = train_model(data)

    st.write("### 📊 Sample Data")
    st.dataframe(data.head())

    st.write(f"### ✅ Model Accuracy: {acc * 100:.2f}%")

    # Classification Report
    st.write("### 🧾 Classification Report")
    report = classification_report(y_test, predictions, output_dict=False)
    st.text(report)

    # Sentiment Prediction
    user_review = st.text_area("Enter a review to predict sentiment:")
    if st.button("Predict Sentiment"):
        result = predict_sentiment(model, user_review)
        st.success(f"Predicted Sentiment: **{result}**")

    # Word Cloud
    if st.checkbox("Show Word Cloud"):
        st.write("### ☁️ Word Cloud of Reviews")
        wc = generate_wordcloud(data, theme)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # Confusion Matrix
    if st.checkbox("Show Confusion Matrix"):
        st.write("### 📉 Confusion Matrix")
        cm = confusion_matrix(y_test, predictions)
        labels = sorted(data['Sentiment'].unique())
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

    # Download Analyzed Data
    st.write("### 📥 Download Analyzed Data")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "analyzed_reviews.csv", "text/csv")

    # Real-Time Review Upload & Prediction
    st.write("### 🔄 Live Review Analysis")
    uploaded_file = st.file_uploader("Upload new reviews (CSV with 'Review' column):", type="csv")

    if uploaded_file is not None:
        if st.button("Refresh Live Reviews"):
            st.experimental_rerun()

        with st.spinner("Analyzing uploaded reviews..."):
            new_data = analyze_uploaded_file(uploaded_file, model)

        if new_data is not None:
            st.write("### 🔍 Predicted Sentiments")
            st.dataframe(new_data)

            # Trend Visualization
            st.write("### 📈 Live Sentiment Distribution")
            sentiment_counts = new_data['Predicted_Sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            ax.axis("equal")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
