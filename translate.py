import os
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from pypdf import PdfReader
from fpdf import FPDF
from tqdm import tqdm
from typing import List
from tempfile import NamedTemporaryFile
from contextlib import contextmanager
from typing import Optional
import argparse
import logging
import torch
import torch.cuda
import arabic_reshaper
from bidi.algorithm import get_display


class PDFExtractionError(Exception):
    # Raised when PDF text extraction fails
    pass


class TranslationError(Exception):
    # Raised when translation fails
    pass


class PDFCreationError(Exception):
    # Raised when creating translated PDF fails
    pass


# Initialize FastAPI app
app = FastAPI()

# Get the directory containing the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)  # Change working directory to script location

# Set fonts dir
FONTS_DIR = os.path.join(BASE_DIR, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

# Then modify your directories setup
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create templates and static directories if they don't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Set up templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Add a context manager for GPU memory handling
@contextmanager
def gpu_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Load model and tokenizer
def load_model():
    try:
        model_name = "facebook/m2m100_1.2B"
        cache_dir = os.path.join(
            BASE_DIR, "model_cache"
        )  # Create a cache directory in your app folder
        tokenizer = M2M100Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = M2M100ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(DEVICE)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# Add model cleanup function
def cleanup_model():
    global MODEL, TOKENIZER
    try:
        if MODEL is not None:
            MODEL.cpu()
            del MODEL
        if TOKENIZER is not None:
            del TOKENIZER
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        MODEL = None
        TOKENIZER = None
    except Exception as e:
        logger.error(f"Error cleaning up model: {e}")


# Global model instance
MODEL = None
TOKENIZER = None
SUPPORTED_LANGUAGES = {"en", "fr", "ar", "es", "zh"}  # Add supported languages
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def get_model() -> (
    tuple[Optional[M2M100ForConditionalGeneration], Optional[M2M100Tokenizer]]
):
    global MODEL, TOKENIZER
    try:
        if MODEL is None or TOKENIZER is None:
            MODEL, TOKENIZER = load_model()
        return MODEL, TOKENIZER
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        cleanup_model()
        raise


def split_text(text: str, max_length: int = 512) -> List[str]:
    # Add logic to split text into manageable chunks
    # This is a simplified version
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_length:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Extract text from PDF
def extract_text_from_pdf(pdf_path: str) -> List[str]:
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        reader = PdfReader(pdf_path)

        # Check if PDF is encrypted
        if reader.is_encrypted:
            raise PDFExtractionError(
                "PDF is encrypted. Please provide an unencrypted PDF."
            )

        if len(reader.pages) == 0:
            raise PDFExtractionError("PDF file is empty")

        text_list = []
        total_text = ""

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                text_list.append(text)
                total_text += text
            except Exception as e:
                logger.error(f"Error extracting text from page {i+1}: {e}")
                text_list.append("")

        # If we got no text at all, the PDF might be scanned
        if not total_text.strip():
            raise PDFExtractionError(
                "No text could be extracted. This might be a scanned PDF or image-based PDF. "
                "Please ensure the PDF contains actual text content."
            )

        return text_list

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except PDFExtractionError as e:
        logger.error(f"PDF extraction error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during PDF extraction: {e}")
        raise PDFExtractionError(f"Failed to extract text from PDF: {str(e)}")


# Translate text
def translate_text(
    text_list: List[str], model, tokenizer, source_lang="en", target_lang="ar"
) -> List[str]:
    if not text_list:
        raise TranslationError("Empty text list provided for translation")

    try:
        translated_texts = []
        with gpu_memory_manager():
            for page_num, text in enumerate(tqdm(text_list, desc="Translating pages")):
                try:
                    if not text.strip():
                        translated_texts.append("")
                        continue

                    chunks = split_text(text)
                    translated_chunks = []

                    for chunk_num, chunk in enumerate(chunks):
                        try:
                            tokenizer.src_lang = source_lang
                            encoded = tokenizer(
                                chunk,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=512,
                            )
                            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

                            with torch.no_grad():
                                generated_tokens = model.generate(
                                    **encoded,
                                    forced_bos_token_id=tokenizer.get_lang_id(
                                        target_lang
                                    ),
                                    max_length=512,
                                )
                                translated_chunk = tokenizer.batch_decode(
                                    generated_tokens, skip_special_tokens=True
                                )[0]
                                translated_chunks.append(translated_chunk)

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            logger.error(
                                f"Error translating chunk {chunk_num} of page {page_num + 1}: {e}"
                            )
                            translated_chunks.append(f"[Translation Error: {str(e)}]")

                    translated_texts.append(" ".join(translated_chunks))

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {e}")
                    translated_texts.append(f"[Page Translation Error: {str(e)}]")

        return translated_texts

    except Exception as e:
        logger.error(f"Fatal error during translation: {e}")
        cleanup_model()
        raise TranslationError(f"Translation process failed: {str(e)}")


def create_translated_pdf(
    original_pdf_path: str, translated_texts: List[str], target_lang: str = "en"
) -> str:
    if not translated_texts:
        raise PDFCreationError("No translated texts provided")

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # For Arabic text
        if target_lang == "ar":
            # Make sure Amiri font exists
            font_path = os.path.join(FONTS_DIR, "Amiri-Regular.ttf")
            if not os.path.exists(font_path):
                raise PDFCreationError(
                    f"Arabic font not found at {font_path}. Please download Amiri font."
                )

            pdf.add_font("Amiri", "", font_path, uni=True)
            pdf.set_font("Amiri", size=14)
        else:
            # For other languages, use default font
            pdf.set_font("Helvetica", size=11)

        for page_num, text in enumerate(translated_texts):
            try:
                pdf.add_page()

                if not text.strip():
                    pdf.cell(0, 10, "[Empty Page]")
                    continue

                lines = text.split("\n")
                for line in lines:
                    if line.strip():
                        if target_lang == "ar":
                            # Reshape Arabic text
                            reshaped_text = arabic_reshaper.reshape(line)
                            # Apply BIDI algorithm
                            bidi_text = get_display(reshaped_text)
                            pdf.set_font("Amiri", size=14)
                            pdf.multi_cell(0, 10, bidi_text, align="R")
                        else:
                            pdf.set_font("Helvetica", size=11)
                            pdf.multi_cell(0, 10, line)

            except Exception as e:
                logger.error(f"Error creating page {page_num + 1}: {e}")
                pdf.add_page()
                pdf.set_font("Helvetica", size=11)
                pdf.cell(0, 10, f"[Error creating page: {str(e)}]")

        output_path = os.path.join(
            os.path.dirname(original_pdf_path),
            f"translated_{os.path.basename(original_pdf_path)}",
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            pdf.output(output_path)
        except Exception as e:
            raise PDFCreationError(f"Failed to save PDF: {str(e)}")

        return output_path

    except Exception as e:
        logger.error(f"Error creating translated PDF: {e}")
        raise PDFCreationError(f"Failed to create translated PDF: {str(e)}")


@app.post("/translate-pdf/")
async def translate_pdf_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_lang: str = "en",
    target_lang: str = "ar",
):

    temp_path = None
    output_path = None

    # Validate inputs
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if source_lang not in SUPPORTED_LANGUAGES or target_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # Check file size
    file_content = await file.read()
    file_size = len(file_content)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        # Create secure temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            temp_file.write(file_content)

        # Process the PDF
        model, tokenizer = get_model()

        try:
            text_list = extract_text_from_pdf(temp_path)
        except PDFExtractionError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "PDF extraction failed",
                    "message": str(e),
                    "suggestion": "Please ensure the PDF contains actual text and is not a scanned document or image-based PDF.",
                },
            )

        try:
            translated_texts = translate_text(
                text_list, model, tokenizer, source_lang, target_lang
            )
        except TranslationError as e:
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

        try:
            output_path = create_translated_pdf(
                temp_path, translated_texts, target_lang
            )
        except PDFCreationError as e:
            raise HTTPException(
                status_code=500, detail=f"PDF creation failed: {str(e)}"
            )

        # Cleanup tasks
        if temp_path:
            background_tasks.add_task(os.unlink, temp_path)
        if output_path:
            background_tasks.add_task(os.unlink, output_path)

        # Return file download response
        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=f"translated_{file.filename}",
        )

    except Exception as e:
        # Cleanup on error
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        cleanup_model()

        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add cleanup on application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    cleanup_model()


def main():
    parser = argparse.ArgumentParser(description="Translate a PDF file.")
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file.")
    parser.add_argument(
        "--source_lang", type=str, default="en", help="Source language code"
    )
    parser.add_argument(
        "--target_lang", type=str, default="ar", help="Target language code"
    )
    args = parser.parse_args()

    try:
        model, tokenizer = load_model()
        text_list = extract_text_from_pdf(args.pdf_path)
        translated_texts = translate_text(
            text_list, model, tokenizer, args.source_lang, args.target_lang
        )
        output_path = create_translated_pdf(args.pdf_path, translated_texts)
        logger.info(f"Translation completed. Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main()
    else:
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
