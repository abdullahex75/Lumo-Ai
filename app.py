
from __future__ import annotations

import os
import json
import base64
import uuid
from pathlib import Path
from io import BytesIO
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from groq import Groq


# =====================================================
# LUMO AI - CONFIGURATION
# =====================================================

APP_NAME = "Lumo AI"
APP_VERSION = "1.0"
CREATOR = "Muhammad Abdullah"

DATA_DIR = Path(__file__).parent / "data"
CHATS_FILE = DATA_DIR / "chats.json"

CHAT_MODELS = {
    "GPT-OSS 120B - Best Quality": "openai/gpt-oss-120b",
    "GPT-OSS 20B - Faster": "openai/gpt-oss-20b",
}

STT_MODEL = "whisper-large-v3-turbo"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

LANGUAGES = [
    "Auto Detect",
    "English",
    "Urdu",
    "Roman Urdu",
    "Hindi",
]

LANGUAGE_RULES = {
    "Auto Detect": "Reply in the same language as the user.",
    "English": "Reply in clear, simple English.",
    "Urdu": "Reply in Urdu script.",
    "Roman Urdu": "Reply in Roman Urdu.",
    "Hindi": "Reply in Hindi script.",
}

MODES = {
    "General Assistant": "You are a helpful and accurate AI assistant.",
    "Study Tutor": "You are a patient tutor. Explain concepts step by step with examples.",
    "Coding Helper": "You are an expert coding assistant. Write correct code and explain it simply.",
    "Writer": "You are a professional writing assistant.",
    "Translator": "You are a careful translator. Preserve meaning and tone.",
    "Summarizer": "Summarize information clearly using short points.",
    "Quiz Maker": "Create educational quizzes with answers and explanations.",
    "Voice Assistant": "Give short, natural conversational answers suitable for speaking aloud.",
}


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title=f"{APP_NAME} | Personal AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================
# PROFESSIONAL DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 0%, #172554 0%, transparent 35%),
        radial-gradient(circle at 90% 5%, #312e81 0%, transparent 30%),
        #080d1b;
    color: #f8fafc;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111c32, #0b1222);
    border-right: 1px solid #263858;
}

h1, h2, h3, p, label {
    color: #f8fafc;
}

.lumo-header {
    text-align: center;
    padding: 32px 15px;
    margin-bottom: 20px;
    border-radius: 22px;
    background: linear-gradient(120deg, #172554, #312e81, #581c87);
    border: 1px solid #5145bd;
    box-shadow: 0 15px 40px rgba(0,0,0,.25);
}

.lumo-header h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.lumo-header p {
    color: #dbeafe;
    margin: 8px 0 0;
}

.stChatMessage {
    background: #111c32;
    border: 1px solid #263858;
    border-radius: 16px;
    padding: 12px;
}

.stButton button, .stDownloadButton button {
    background: linear-gradient(135deg, #5145bd, #6d5ce8);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 40px;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #8175f5;
    color: white;
}

[data-testid="stChatInput"] {
    background: #111c32;
    border: 1px solid #344766;
    border-radius: 14px;
}

.badge {
    display: inline-block;
    padding: 5px 11px;
    margin: 3px;
    border-radius: 20px;
    background: #1e2b4a;
    border: 1px solid #34487a;
    color: #dbeafe;
    font-size: 12px;
}

.stat-box {
    background: #111c32;
    border: 1px solid #263858;
    border-radius: 12px;
    padding: 13px;
    text-align: center;
}

div[data-testid="stExpander"] {
    border: 1px solid #263858;
    border-radius: 12px;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .lumo-header h1 {
        font-size: 32px;
    }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# CHAT STORAGE
# =====================================================

def load_chats():
    try:
        if CHATS_FILE.exists():
            data = json.loads(CHATS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        st.warning("Saved chat history could not be loaded.")
    return {}


def save_chats():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHATS_FILE.write_text(
            json.dumps(st.session_state.chats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as error:
        st.error(f"Could not save chats: {error}")


def create_chat():
    chat_id = uuid.uuid4().hex[:10]

    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "created": datetime.now().strftime("%d %b %Y, %H:%M"),
        "messages": [],
        "document": None,
    }

    st.session_state.current_chat = chat_id
    save_chats()


def initialize_state():
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()

        if not st.session_state.chats:
            create_chat()
        else:
            st.session_state.current_chat = list(
                st.session_state.chats.keys()
            )[-1]

    st.session_state.setdefault("manual_key", "")
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("voice_draft", "")
    st.session_state.setdefault("audio_counter", 0)


def get_current_chat():
    return st.session_state.chats[
        st.session_state.current_chat
    ]


# =====================================================
# GROQ API
# =====================================================

def get_api_key():
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    return (
        secret_key
        or os.getenv("GROQ_API_KEY", "")
        or st.session_state.manual_key
    )


def get_client():
    key = get_api_key()

    if not key:
        return None

    return Groq(api_key=key)


# =====================================================
# DOCUMENT READER
# =====================================================

def read_document(upload):
    filename = upload.name.lower()
    data = upload.getvalue()

    if filename.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(data))
        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    else:
        text = data.decode("utf-8", errors="replace")

    return text.strip()[:15000]


# =====================================================
# AI RESPONSE
# =====================================================

def build_system_prompt(language, mode, user_name, document):
    prompt = f"""
You are {APP_NAME}, a professional personal AI assistant.

Creator: {CREATOR}

Your role:
{MODES[mode]}

Language instruction:
{LANGUAGE_RULES[language]}

Be helpful, accurate, respectful and clear.
If you are unsure, say so honestly.
Explain difficult topics in simple language.
"""

    if user_name.strip():
        prompt += f"\nThe user's preferred name is {user_name.strip()}."

    if document:
        prompt += f"""

The user attached a document named: {document['name']}

Use this document as context when answering questions.
If the answer is not in the document, say so.

DOCUMENT CONTENT:
{document['text']}
"""

    return prompt


def generate_answer(chat, language, mode, user_name, model):
    client = get_client()

    if client is None:
        st.error("Groq API key is missing. Add it in the sidebar.")
        return None

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                language,
                mode,
                user_name,
                chat.get("document"),
            ),
        }
    ]

    # Keep recent conversation messages.
    for message in chat["messages"][-16:]:
        messages.append({
            "role": message["role"],
            "content": message["content"],
        })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=2000,
        stream=True,
    )

    return st.write_stream(
        chunk.choices[0].delta.content
        for chunk in response
        if chunk.choices and chunk.choices[0].delta.content
    )


# =====================================================
# VOICE TO TEXT
# =====================================================

def transcribe_audio(audio, language):
    client = get_client()

    if client is None:
        raise ValueError("Please add your Groq API key first.")

    options = {
        "file": (
            audio.name or "voice.wav",
            audio.getvalue(),
        ),
        "model": STT_MODEL,
        "response_format": "json",
    }

    language_codes = {
        "English": "en",
        "Urdu": "ur",
        "Roman Urdu": "ur",
        "Hindi": "hi",
    }

    if language in language_codes:
        options["language"] = language_codes[language]

    result = client.audio.transcriptions.create(**options)
    return result.text.strip()


# =====================================================
# TEXT TO VOICE
# =====================================================

def make_speech(text, language):
    from gtts import gTTS

    language_codes = {
        "Urdu": "ur",
        "Hindi": "hi",
        "Roman Urdu": "en",
        "English": "en",
        "Auto Detect": "en",
    }

    speech_language = language_codes.get(language, "en")

    audio_buffer = BytesIO()

    gTTS(
        text=text[:2000],
        lang=speech_language,
    ).write_to_fp(audio_buffer)

    audio_buffer.seek(0)
    return audio_buffer


# =====================================================
# IMAGE ANALYSIS
# =====================================================

def analyze_image(upload, question, language):
    client = get_client()

    if client is None:
        raise ValueError("Please add your Groq API key first.")

    encoded_image = base64.b64encode(
        upload.getvalue()
    ).decode("utf-8")

    mime_type = upload.type or "image/png"

    result = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful image analysis assistant. "
                    + LANGUAGE_RULES[language]
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question or "Describe this image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=1200,
    )

    return result.choices[0].message.content or ""


# =====================================================
# SIDEBAR
# =====================================================

def render_sidebar():
    with st.sidebar:
        st.title("✨ Lumo AI")
        st.caption("YOUR PERSONAL AI ASSISTANT")

        st.divider()

        if not get_api_key():
            st.warning("Groq API key is not configured.")
            st.session_state.manual_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Your key is used locally for this app session.",
            )

        if st.button("➕ New Chat", use_container_width=True):
            create_chat()
            st.rerun()

        st.divider()
        st.markdown("### 💬 Chat History")

        for chat_id, saved_chat in reversed(
            list(st.session_state.chats.items())
        ):
            title = saved_chat["title"][:25]

            if st.button(
                title,
                key=f"select_{chat_id}",
                use_container_width=True,
            ):
                st.session_state.current_chat = chat_id
                st.rerun()

        st.divider()

        st.markdown("### 🎛️ Assistant Settings")

        user_name = st.text_input(
            "Your name",
            value="",
            placeholder="Muhammad Abdullah",
        )

        language = st.selectbox(
            "Response Language",
            LANGUAGES,
        )

        mode = st.selectbox(
            "Assistant Mode",
            list(MODES.keys()),
        )

        model_label = st.selectbox(
            "AI Model",
            list(CHAT_MODELS.keys()),
        )

        st.divider()

        st.markdown("### 🔊 Voice Settings")

        speak_answers = st.toggle(
            "Enable text-to-voice",
            value=False,
        )

        st.divider()

        st.markdown("### 📎 Documents")

        chat = get_current_chat()

        upload = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            key=f"document_{st.session_state.current_chat}",
        )

        if upload and st.button(
            "Attach Document",
            use_container_width=True,
        ):
            try:
                extracted = read_document(upload)

                if not extracted:
                    st.warning("No readable text found.")
                else:
                    chat["document"] = {
                        "name": upload.name,
                        "text": extracted,
                    }
                    save_chats()
                    st.success("Document attached!")
                    st.rerun()

            except Exception as error:
                st.error(f"Document error: {error}")

        if chat.get("document"):
            st.caption(
                f"Attached: {chat['document']['name']}"
            )

            if st.button("Remove Document"):
                chat["document"] = None
                save_chats()
                st.rerun()

        st.divider()

        if chat["messages"]:
            export_text = "\n\n".join(
                f"{m['role'].upper()}:\n{m['content']}"
                for m in chat["messages"]
            )

            st.download_button(
                "⬇️ Export Chat",
                data=export_text,
                file_name="lumo_chat.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("🧹 Clear Current Chat", use_container_width=True):
            chat["messages"] = []
            chat["title"] = "New Chat"
            save_chats()
            st.rerun()

        st.divider()

        st.caption(f"Created by {CREATOR}")
        st.caption(f"{APP_NAME} v{APP_VERSION}")

    return (
        user_name,
        language,
        mode,
        CHAT_MODELS[model_label],
        speak_answers,
    )


# =====================================================
# MAIN HEADER
# =====================================================

def render_header(language, mode):
    st.markdown(
        f"""
        <div class="lumo-header">
            <h1>✨ Lumo AI</h1>
            <p>Your Personal Smart AI Assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <span class="badge">🤖 {mode}</span>
        <span class="badge">🌐 {language}</span>
        <span class="badge">⚡ Groq AI</span>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# WELCOME SCREEN
# =====================================================

def render_welcome():
    st.markdown("## 👋 Welcome to Lumo AI")
    st.write("What would you like help with today?")

    prompts = [
        "Explain artificial intelligence in simple words.",
        "Help me learn Python programming.",
        "Explain photosynthesis step by step.",
        "Make a short quiz about science.",
    ]

    columns = st.columns(2)

    for index, prompt in enumerate(prompts):
        if columns[index % 2].button(
            prompt,
            key=f"welcome_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_prompt = prompt
            st.rerun()


# =====================================================
# CHAT HISTORY DISPLAY
# =====================================================

def render_history(chat):
    for index, message in enumerate(chat["messages"]):
        role = message["role"]

        avatar = "🧑" if role == "user" else "✨"

        with st.chat_message(role, avatar=avatar):
            st.markdown(message["content"])

            if role == "assistant":
                if st.button(
                    "🔊 Listen",
                    key=f"listen_{st.session_state.current_chat}_{index}",
                ):
                    try:
                        audio = make_speech(
                            message["content"],
                            st.session_state.selected_language,
                        )
                        st.audio(audio, format="audio/mp3")
                    except Exception as error:
                        st.error(f"Voice error: {error}")


# =====================================================
# VOICE INPUT
# =====================================================

def render_voice_input(language):
    with st.expander("🎙️ Talk to Lumo AI"):
        audio = st.audio_input(
            "Record your voice",
            key=f"audio_{st.session_state.audio_counter}",
        )

        if st.button("Convert Voice to Text"):
            if audio is None:
                st.warning("Please record your voice first.")
            else:
                try:
                    with st.spinner("Understanding your voice..."):
                        recognized = transcribe_audio(audio, language)

                    if recognized:
                        st.session_state.pending_prompt = recognized
                        st.session_state.audio_counter += 1
                        st.rerun()
                    else:
                        st.warning("No speech was recognized.")

                except Exception as error:
                    st.error(f"Voice recognition failed: {error}")


# =====================================================
# IMAGE UPLOAD
# =====================================================

def render_image_tool(language, chat):
    with st.expander("🖼️ Analyze an Image"):
        image = st.file_uploader(
            "Upload PNG, JPG or WEBP",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_upload",
        )

        question = st.text_input(
            "What would you like to know about the image?",
            placeholder="Describe this image...",
        )

        if st.button("Analyze Image"):
            if image is None:
                st.warning("Please upload an image.")
                return

            if len(image.getvalue()) > 3 * 1024 * 1024:
                st.warning("Image must be smaller than 3 MB.")
                return

            try:
                with st.spinner("Analyzing image..."):
                    answer = analyze_image(
                        image,
                        question,
                        language,
                    )

                chat["messages"].append({
                    "role": "user",
                    "content": f"Image question: {question or 'Describe this image.'}",
                })

                chat["messages"].append({
                    "role": "assistant",
                    "content": answer,
                })

                save_chats()
                st.rerun()

            except Exception as error:
                st.error("Image analysis failed.")
                st.caption(str(error))


# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    initialize_state()

    (
        user_name,
        language,
        mode,
        model,
        speak_answers,
    ) = render_sidebar()

    st.session_state.selected_language = language

    chat = get_current_chat()

    render_header(language, mode)

    if not chat["messages"] and not st.session_state.pending_prompt:
        render_welcome()

    render_voice_input(language)
    render_image_tool(language, chat)
    render_history(chat)

    typed_prompt = st.chat_input(
        "Message Lumo AI..."
    )

    prompt = (
        st.session_state.pending_prompt
        or typed_prompt
    )

    st.session_state.pending_prompt = None

    if prompt:
        prompt = prompt.strip()[:6000]

        if prompt:
            chat["messages"].append({
                "role": "user",
                "content": prompt,
            })

            if chat["title"] == "New Chat":
                chat["title"] = prompt[:35]

            save_chats()

            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="✨"):
                try:
                    with st.spinner("Lumo AI is thinking..."):
                        answer = generate_answer(
                            chat,
                            language,
                            mode,
                            user_name,
                            model,
                        )

                    if answer:
                        chat["messages"].append({
                            "role": "assistant",
                            "content": answer,
                        })

                        save_chats()

                        if speak_answers:
                            try:
                                audio = make_speech(answer, language)
                                st.audio(audio, format="audio/mp3")
                            except Exception as voice_error:
                                st.warning(
                                    f"Voice playback unavailable: {voice_error}"
                                )

                except Exception as error:
                    st.error("Lumo AI could not generate a reply.")
                    st.caption(str(error))

    if chat["messages"]:
        st.divider()

        total_messages = len(chat["messages"])
        total_words = sum(
            len(message["content"].split())
            for message in chat["messages"]
        )

        c1, c2 = st.columns(2)

        c1.metric("Total Messages", total_messages)
        c2.metric("Total Words", total_words)

    st.divider()

    st.markdown(
        f"""
        <div style="text-align:center;color:#94a3b8;padding:12px;">
            ✨ {APP_NAME} v{APP_VERSION}
            <br>
            Created by {CREATOR}
            <br>
            Powered by Groq
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
