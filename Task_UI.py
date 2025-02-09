import re
import whisper
import gradio as gr
from transformers import pipeline


def transcribe_file(file_path: str) -> dict:
    # Load the Whisper model (using the "base" model)
    model = whisper.load_model("base")
    return model.transcribe(file_path, word_timestamps=True)


def annotate_segments(transcription: dict) -> list:
    segments = transcription.get("segments", [])
    annotated = []
    speaker = 1
    for seg in segments:
        annotated.append({
            "speaker": f"Speaker {speaker}",
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text").strip()
        })
        # Alternate speakers for demonstration purposes
        speaker = 2 if speaker == 1 else 1
    return annotated


def summarize(text: str) -> str:
    # Explicitly specifying the summarization model and revision
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e"
    )
    summary = summarizer(text, max_length=250, min_length=70, do_sample=False)
    return summary[0]['summary_text']


def analyze_content(text: str) -> (int, list):
    # Count words using a simple regex
    words = re.findall(r'\w+', text)
    word_count = len(words)

    # Explicitly specifying the sentiment-analysis model and revision
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"
    )
    sentiment = sentiment_analyzer(text)
    return word_count, sentiment


def analyze_audio(audio_file):
    # Transcribe the audio file using Whisper
    transcript = transcribe_file(audio_file)
    full_text = transcript.get("text", "")

    # Annotate each segment with speaker labels and timestamps
    segments = annotate_segments(transcript)
    detailed_transcript = "\n".join(
        [f"[{seg['speaker']}] ({seg['start']:.2f}-{seg['end']:.2f}): {seg['text']}" for seg in segments]
    )

    # Generate a summary of the full transcription
    summary = summarize(full_text)

    # Analyze the transcription content for word count and sentiment
    word_count, sentiment = analyze_content(full_text)
    analysis = f"Word Count: {word_count}\nSentiment: {sentiment}"

    return detailed_transcript, summary, analysis


# Set up the Gradio interface
interface = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Enhanced Transcription"),
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Analysis")
    ],
    title="Audio Analysis",
    description="Upload an audio file to transcribe, summarize, and analyze it."
)

if __name__ == "__main__":
    interface.launch()
