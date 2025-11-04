"""Table parsing service for converting image tables to structured JSON."""

import os
import re
from typing import Dict, Any, Union

import pytesseract
from PIL import Image
import camelot
import pandas as pd
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


class MedicalTableParser:
    """Parser for extracting medical data from table images or PDFs."""

    def __init__(self, use_advanced_model: bool = False):
        """
        Initialize parser with chosen model.

        Args:
            use_advanced_model: If True, use Donut model for complex images.
        """
        self.use_advanced_model = use_advanced_model

        self.field_definitions = {
            "Pregnancies": {"patterns": ["pregnancies"], "range": (0, 20)},
            "Glucose": {"patterns": ["glucose"], "range": (0, 500)},
            "BloodPressure": {"patterns": ["bloodpressure", "blood pressure"], "range": (0, 300)},
            "SkinThickness": {"patterns": ["skinthickness", "skin thickness"], "range": (0, 100)},
            "Insulin": {"patterns": ["insulin"], "range": (0, 1000)},
            "BMI": {"patterns": ["bmi"], "range": (0, 100)},
            "DiabetesPedigreeFunction": {"patterns": ["diabetespedigreefunction", "pedigree"], "range": (0, 3)},
            "Age": {"patterns": ["age"], "range": (0, 120)},
            "Outcome": {"patterns": ["outcome"], "range": (0, 1)}
        }

        # Initialize Donut model if requested
        if self.use_advanced_model:
            print("Loading Donut model for advanced parsing...")
            self.processor = DonutProcessor.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-docvqa"
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                "naver-clova-ix/donut-base-finetuned-docvqa"
            )
            if torch.cuda.is_available():
                self.model.to("cuda")

    # -------------------- Public API -------------------- #

    def parse_image(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """Parse a table from an image path or PIL Image."""
        try:
            if isinstance(image_input, Image.Image):
                image = image_input
                ext = ".png"
            else:
                ext = os.path.splitext(image_input)[1].lower()
                image = Image.open(image_input)

            if ext == ".pdf":
                return self._parse_pdf(image_input)
            elif self.use_advanced_model:
                return self._parse_with_donut(image_input if isinstance(image_input, str) else image)
            else:
                return self._parse_with_tesseract(image_input if isinstance(image_input, str) else image)
        except Exception as e:
            print(f"Error parsing {image_input}: {e}")
            return self._empty_result()

    # -------------------- Internal Helpers -------------------- #

    def _empty_result(self) -> Dict[str, Any]:
        """Return a dict with all fields set to None."""
        return {field: None for field in self.field_definitions}

    def _parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a table from a PDF using Camelot."""
        try:
            tables = camelot.read_pdf(pdf_path, pages="1")
            if not tables:
                raise ValueError("No tables found in PDF")
            df = tables[0].df
            return self._extract_fields_from_df(df)
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return self._empty_result()

    def _parse_with_tesseract(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """Parse an image table using Tesseract OCR."""
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image path not found: {image_input}")
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise TypeError(f"Invalid image input type: {type(image_input)}")

  
            text = pytesseract.image_to_string(image, config="--psm 6")
            lower_text = text.lower()

            print(f"\n===== OCR OUTPUT =====\n{text}\n=======================\n")

           
            extracted = self._extract_fields_from_text(lower_text)
            if any(value is not None for value in extracted.values()):
                print("Extracted biomarkers directly from OCR text.")
                return extracted

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) < 2:
                print("Not enough lines detected by OCR.")
                return extracted or self._empty_result()

            data = [re.split(r"\s+", line) for line in lines]
            df = pd.DataFrame(data[1:], columns=data[0]) if len(data) > 1 else pd.DataFrame()

            if df.empty:
                print("DataFrame is empty after parsing OCR output.")
                return extracted or self._empty_result()

            print("\n===== PARSED DATAFRAME =====")
            print(df)
            print("============================\n")

            return self._extract_fields_from_df(df)

        except Exception as e:
            print(f"Tesseract parsing error: {e}")
            return self._empty_result()

    def _parse_with_donut(self, image_path: str) -> Dict[str, Any]:
        """Parse a table using Donut model."""
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to("cuda")

            outputs = self.model.generate(
                pixel_values,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            return self._extract_fields_from_text(text)
        except Exception as e:
            print(f"Donut parsing error: {e}")
            return self._empty_result()

    def _extract_fields_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract and normalize fields from a DataFrame."""
        result = self._empty_result()
        for field, meta in self.field_definitions.items():
            for pattern in meta["patterns"]:
                matches = [col for col in df.columns if re.search(pattern, col.lower())]
                if matches:
                    try:
                        value = float(df[matches[0]].iloc[0])
                        if meta["range"][0] <= value <= meta["range"][1]:
                            result[field] = value
                    except:
                        pass
                    break
        return result

    def _extract_fields_from_text(self, text: str) -> Dict[str, Any]:
        """Extract fields directly from raw text."""
        result = self._empty_result()
        for field, meta in self.field_definitions.items():
            for pattern in meta["patterns"]:
                match = re.search(rf"{pattern}[:\s=\-\+]*([\d\.]+)", text)
                if match:
                    try:
                        value = float(match.group(1))
                        if meta["range"][0] <= value <= meta["range"][1]:
                            result[field] = value
                    except ValueError:
                        pass
                    break
        return result
