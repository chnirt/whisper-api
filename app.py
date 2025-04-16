from fastapi import FastAPI, File, UploadFile
import whisper
import uvicorn

app = FastAPI()

# Load the Whisper model - options: tiny, base, small, medium, large
model = whisper.load_model("small")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read the uploaded audio file contents
    contents = await file.read()
    
    # Save the uploaded file temporarily
    file_path = "temp_audio"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Use Whisper to transcribe the audio file
    result = model.transcribe(file_path)
    
    # Return the transcribed text as JSON
    return {"transcription": result["text"]}

if __name__ == "__main__":
    # Run the FastAPI application with uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
