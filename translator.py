import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def check_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"Error by loadin model {model_name}: {e}")

check_model_and_tokenizer("Helsinki-NLP/opus-mt-en-de")

@st.cache_resource
def load_model():
    return pipeline("translation_en_to_uk", model="Helsinki-NLP/opus-mt-en-de")

translator = load_model()

st.title("Online Translate")
st.subheader("Translate from english to german")

input_text = st.text_area("Enter your text for translate", height=200)

if st.button("Translate"):
    if input_text.strip():
        translation = translator(input_text)[0]['translation_text']
        st.success("Translation:")
        st.write(translation)
    else:
        st.warning("Please enter your text.")
