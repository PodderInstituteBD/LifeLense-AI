python
"""
OCR engine for text extraction from images.
Uses free Tesseract OCR.
"""

import pytesseract
from PIL import Image
from io import BytesIO
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class OCREngine:
    """Handle OCR for images"""
    
    @staticmethod
    def preprocess_image(image_data: bytes) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        try:
            # Load image
            img = Image.open(BytesIO(image_data))
            img_array = np.array(img)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Resize if too small
            height, width = img_array.shape
            if height < 300 or width < 300:
                scale = max(300 / height, 300 / width)
                img_array = cv2.resize(
                    img_array, 
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_CUBIC
                )
            
            # Apply contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            
            # Apply thresholding
            _, img_array = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return original as fallback
            img = Image.open(BytesIO(image_data))
            return np.array(img)
    
    @staticmethod
    def extract_text(image_data: bytes) -> Dict:
        """Extract text from image using Tesseract OCR"""
        try:
            # Preprocess image
            processed_img = OCREngine.preprocess_image(image_data)
            
            # Extract text
            extracted_text = pytesseract.image_to_string(processed_img)
            
            # Get confidence data
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            confidence = np.mean([int(c) for c in data['confidence'] if int(c) > 0]) / 100
            
            return {
                "text": extracted_text.strip(),
                "confidence": round(confidence, 2),
                "words_found": len(extracted_text.split())
            }
            
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract is not installed")
            return {
                "text": "",
                "confidence": 0,
                "error": "Tesseract OCR is not installed. Please install it to enable text extraction."
            }
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return {
                "text": "",
                "confidence": 0,
                "error": f"Failed to extract text: {str(e)}"
            }


ocr_engine = OCREngine()
