import streamlit as st
from youtube_processor import extract_video_id, get_transcript, setup_qa_chain


st.set_page_config(
    page_title="YouTube Video Q&A",
    page_icon="▶️",
    layout="centered"
)


if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "video_info" not in st.session_state:
    st.session_state.video_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("YouTube Video Q&A")
st.markdown("Ask questions about any YouTube video's content")


video_url = st.text_input(
    "Enter YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

if video_url:
    if st.session_state.qa_chain is None:
        with st.spinner("Processing video transcript..."):
            try:
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL. Please enter a valid link.")
                    st.stop()
                
          
                transcript = get_transcript(video_id)
                st.session_state.qa_chain = setup_qa_chain(transcript)  # No API key needed here
                
            
                st.session_state.video_info = {
                    "id": video_id,
                    "url": f"https://youtu.be/{video_id}"
                }
                st.success("Video processed! Ask your questions below.")
                
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.stop()


if st.session_state.qa_chain:

    if st.session_state.video_info:
        st.video(st.session_state.video_info["url"])
    

    question = st.chat_input("Ask about the video...")
    
    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        

        with st.spinner("Finding answer..."):
            try:
                answer = st.session_state.qa_chain.invoke(question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                })
        

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])