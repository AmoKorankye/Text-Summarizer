import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import streamlit as st

# Text Summarizer logo
col1, col2, col3 = st.columns(3)
with col2:
    st.image("text summary logo.png")

file = st.text_input("Paste the text to be summarized: ")


if file:  # Check if file is not empty
    st.header("Text to be paraphrased:")
    st.write(file)
    
    # Read the text to be summarized
    def read_article(file):
        article = file.split(". ")
        sentences = []
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop() 
        
        return sentences

    # Find the similarity between sentences
    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    # Build similarity matrix
    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: #ignore if both are same sentences
                    continue 
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix

    def generate_summary(file, top_n=5):
        nltk.download("stopwords")
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences =  read_article(file)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize text   
        text = ". ".join(summarize_text)
        
        return text

    summary = generate_summary(file, 2)
    st.header("Paraphrased Text:")
    st.write(summary)
else:
    st.write("Please input some text to summarize.")