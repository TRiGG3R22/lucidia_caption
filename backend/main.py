import os
import io
import re
import threading
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Visionary ML Backend")

# Enable CORS so your Next.js frontend on localhost:3000 can communicate with localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Config -------------------------------------------------------------

MAX_FILE_SIZE_MB = 8
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

# --- Local model cache (thread-safe lazy load) ---------------------------

LOCAL_MODEL = None
LOCAL_PROCESSOR = None
_MODEL_LOCK = threading.Lock()


def get_local_model():
    """Lazy load the local BLIP model if the user wants offline execution.

    Thread-safe: guards against two concurrent requests both trying to
    load the (large) model into memory at the same time.
    """
    global LOCAL_MODEL, LOCAL_PROCESSOR
    if LOCAL_MODEL is not None and LOCAL_PROCESSOR is not None:
        return LOCAL_PROCESSOR, LOCAL_MODEL

    with _MODEL_LOCK:
        # Re-check after acquiring the lock in case another thread
        # finished loading while we were waiting.
        if LOCAL_MODEL is None or LOCAL_PROCESSOR is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                print("Loading local BLIP model into memory... (This might take a moment the first time)")
                LOCAL_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                LOCAL_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                print("Local BLIP model loaded successfully!")
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="Transformers/PyTorch not installed correctly. Please install required dependencies to run locally."
                )
    return LOCAL_PROCESSOR, LOCAL_MODEL


def query_huggingface_api(image_bytes: bytes, hf_token: str) -> str:
    """Sends image to Hugging Face Inference API for cloud-mode captioning."""
    api_url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=30)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not reach Hugging Face API: {str(e)}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Hugging Face API Error: {response.text}"
        )

    try:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse HF API response: {str(e)}")


class LLMCaptionError(Exception):
    """Raised when the LLM styling step fails, so callers can decide
    whether to fall back rather than surface a raw error as if it were
    a real caption."""
    pass


def query_llm_for_social_caption(base_caption: str, llm_token: str) -> str:
    """Uses Groq (Llama 3) to rewrite the plain description into an
    engaging Instagram caption. Raises LLMCaptionError on any failure
    instead of returning an error string as if it were content."""
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_token}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an elite, highly creative social media manager. "
        "Take the provided literal description of an image and write ONE highly engaging, "
        "personalized Instagram caption. Add 1-2 expressive emojis and exactly 3 relevant trending hashtags. "
        "Keep it concise, punchy, and modern. Do not output any thinking or meta-text."
    )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Literal description: {base_caption}"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch LLM caption: {str(e)}")
        raise LLMCaptionError(str(e))

    if response.status_code != 200:
        print(f"LLM API Error: {response.text}")
        raise LLMCaptionError(f"Status code {response.status_code}: {response.text}")

    try:
        return response.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as e:
        print(f"Failed to parse LLM response: {str(e)}")
        raise LLMCaptionError(f"Malformed response: {str(e)}")


def fallback_mock_instagram_caption(base_caption: str) -> str:
    """Generates a clever local/mock caption if no LLM key is supplied
    or the LLM call fails."""
    cleaned = base_caption.strip().rstrip('.')
    words = cleaned.split()
    # Strip non-alphanumeric characters so the hashtag is always valid
    last_word = re.sub(r"[^A-Za-z0-9]", "", words[-1]) if words else ""
    hashtag = last_word if last_word else "AILife"
    return f"Vibe check: {cleaned}! ✨📸 Just living in the moment. #PhotoOfTheDay #Vibes #{hashtag}"


@app.post("/api/generate-caption")
async def generate_caption(
    file: UploadFile = File(...),
    use_cloud: bool = Form(True),
    x_hf_token: str = Header(None, alias="X-HF-Token"),
    x_llm_token: str = Header(None, alias="X-LLM-Token")
):
    """
    Main endpoint. Takes an image file and options to route either to
    local Hugging Face transformers or Hugging Face cloud APIs.
    """
    try:
        # 1. Validate content-type up front (belt-and-suspenders; PIL
        #    validation below is the real gatekeeper)
        if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{file.content_type}'. Please upload a JPEG, PNG, WEBP, GIF, or BMP image."
            )

        # 2. Read image bytes, enforcing a size cap so a huge upload
        #    can't blow up server memory.
        image_content = await file.read()
        if len(image_content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB}MB."
            )
        if not image_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Validate that we have a real, decodable image
        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

        base_caption = ""

        # 3. Extract base caption (Cloud HF API vs Local Transformers)
        if use_cloud:
            # Fallback to local env key if frontend header isn't passed
            token = x_hf_token or os.getenv("HF_TOKEN")
            if not token:
                raise HTTPException(
                    status_code=400,
                    detail="Hugging Face API Token missing. Please add it to your environment or input it in settings."
                )
            base_caption = query_huggingface_api(image_content, token)
        else:
            # Local Inference
            processor, model = get_local_model()
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            base_caption = processor.decode(out[0], skip_special_tokens=True)

        # 4. Apply Instagram styling with LLM, gracefully falling back
        #    to the local mock styling if no key is set or the call fails.
        llm_token = x_llm_token or os.getenv("GROQ_API_KEY")
        if llm_token:
            try:
                instagram_caption = query_llm_for_social_caption(base_caption, llm_token)
            except LLMCaptionError:
                instagram_caption = fallback_mock_instagram_caption(base_caption)
        else:
            instagram_caption = fallback_mock_instagram_caption(base_caption)

        return JSONResponse({
            "status": "success",
            "data": {
                "base_caption": base_caption.capitalize(),
                "instagram_caption": instagram_caption
            }
        })

    except HTTPException:
        # Already a well-formed error response; let it propagate as-is.
        raise
    except Exception as e:
        print(f"Exception during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
