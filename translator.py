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
def load_model(model_name, task):
    return pipeline(task, model=model_name)

translator_de = load_model("Helsinki-NLP/opus-mt-en-de", "translation_en_to_de")
translator_fr = load_model("Helsinki-NLP/opus-mt-en-fr", "translation_en_to_fr")

st.title("Online Translate")

st.subheader("Translate from English to German")
input_text_de = st.text_area("Enter your text for translation", height=200, key="de_translation_input")

if st.button("Translate to German"):
    if input_text_de.strip():
        translation_de_result = translator_de(input_text_de)[0]['translation_text']
        st.success("Translation to German:")
        st.write(translation_de_result)
    else:
        st.warning("Please enter your text.")

# Second translation window (English to French)
st.subheader("Translate from English to French")
input_text_fr = st.text_area("Enter your text for translation", height=200, key="fr_translation_input")

if st.button("Translate to French"):
    if input_text_fr.strip():
        translation_fr_result = translator_fr(input_text_fr)[0]['translation_text']
        st.success("Translation to French:")
        st.write(translation_fr_result)
    else:
        st.warning("Please enter your text.")
