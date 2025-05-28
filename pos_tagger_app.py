import pickle
import streamlit as st

# Load the pre-trained HMM model
with open("hmm_model.pkl", "rb") as f:
    transition_probs, emission_probs, start_probs, all_tags = pickle.load(f)

# Viterbi Algorithm
def viterbi(words_seq):
    V = [{}]
    path = {}

    for tag in all_tags:
        emission = emission_probs.get(tag, {}).get(words_seq[0], 1e-6)
        V[0][tag] = start_probs.get(tag, 1e-6) * emission
        path[tag] = [tag]

    for t in range(1, len(words_seq)):
        V.append({})
        new_path = {}

        for curr_tag in all_tags:
            (prob, prev_tag) = max(
                (V[t - 1][pt] *
                 transition_probs.get(pt, {}).get(curr_tag, 1e-6) *
                 emission_probs.get(curr_tag, {}).get(words_seq[t], 1e-6), pt)
                for pt in all_tags
            )
            V[t][curr_tag] = prob
            new_path[curr_tag] = path[prev_tag] + [curr_tag]

        path = new_path

    (prob, final_tag) = max((V[-1][tag], tag) for tag in all_tags)
    return path[final_tag]

# ----------------------
# Streamlit App UI
# ----------------------

st.set_page_config(page_title="HMM POS Tagger", page_icon="🧠", layout="centered")

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Instructions", "About"])

# ----------------------
# Pages
# ----------------------

if page == "Home":
    st.title("🧠 POS Tagger using HMM")
    st.markdown("""
    This app uses a **Hidden Markov Model (HMM)** trained on the Brown corpus to perform Part-of-Speech (POS) tagging.
    
    Simply enter a sentence below, and it will predict the most likely POS tags using the **Viterbi algorithm**.
    """)

    st.subheader("🔤 Enter your sentence:")
    user_input = st.text_input("")

    if user_input:
        tokens = [word.lower() for word in user_input.split()]
        tags = viterbi(tokens)

        st.subheader("🧾 POS Tagging Result:")
        st.markdown("---")

        for word, tag in zip(tokens, tags):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<span style='font-weight:bold;font-size:18px'>{word}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:gray;font-size:16px'>{tag}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.success("✅ POS tagging completed!")
    else:
        st.info("Please enter a sentence to get started.")

elif page == "Instructions":
    st.title("📘 Instructions")
    st.markdown("""
    1. Go to the **Home** page using the sidebar.
    2. Enter a sentence in the input box.
    3. The app will display the most likely POS tags using the HMM model.

    **Example Input:** `The quick brown fox jumps over the lazy dog`
    """)

elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    This is a simple POS tagger built with:
    - Hidden Markov Model (HMM)
    - Viterbi Algorithm
    - Brown Corpus
    - Streamlit for the interface

     
    **Libraries Used:** Streamlit, NLTK, Pickle
    """)

# Footer
st.markdown("<small>Built  using Streamlit & NLTK</small>", unsafe_allow_html=True)
