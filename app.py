import os
import validators
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the LLM model (yaha model initialize karo)
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Streamlit APP Configuration
st.set_page_config(page_title="Chat-Mate...Summarize Text From YT or Website", page_icon="🦜", layout="centered")

# Page Title and Subtitle
st.title("Chat-Mate...Summarize Text From YT or Website 📝")
st.subheader('Summarize any URL with ease')

# URL input and language selection
generic_url = st.text_input("Enter the URL (YouTube video or website)")
selected_language = st.selectbox("Choose Transcript Language", ["English (en)", "Hindi (hi)"])  # Language options

# Extract the language code from the selected option
selected_language_code = selected_language.split(" ")[-1].strip("()")

# Template for summary prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarize Button
if st.button("Summarize Now"):
    # Validate inputs (inputs validate karo)
    if not groq_api_key:
        st.error("API key not found! Please check your .env file.")
    elif not generic_url.strip():
        st.error("Please enter a URL to proceed.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Load data from YouTube or website (Data load karo)
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True, language=selected_language_code)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Provide a dropdown to view the extracted content (Extracted content dekhne ka option)
                if docs:
                    with st.expander("View Extracted Content"):
                        st.write(docs[0].page_content)
                else:
                    st.error("No content could be extracted from the provided URL.")

                # Summarization chain (Summarization chain chalaye)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success("Summary:")
                st.write(output_summary)

                # Download the summary as a text file (Summary download karo)
                if output_summary:
                    st.download_button(
                        label="Download Summary",
                        data=output_summary,
                        file_name="summary.txt",
                        mime="text/plain",
                    )
        except Exception as e:
            st.exception(f"Exception: {e}")