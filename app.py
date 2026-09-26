"""Receipt and invoice extractor demo.

Run with:
    streamlit run app.py
"""

import io
import json
import os
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")


# Prompt for Gemini
EXTRACTION_PROMPT = """
You are a careful receipt and invoice data extraction system.

Inspect the attached image and return ONLY one valid JSON object. Do not use
Markdown fences, explanations, comments, or any text before or after the JSON.

Extract what is visibly printed or handwritten. Never guess. If a value is
missing, unreadable, ambiguous, or not confidently supported by the image,
return null for that value. Preserve the receipt's apparent currency symbol or
currency code when one is visible. If no currency is visible, use null.

Use exactly this structure:
{
  "merchant_name": string or null,
  "date": string or null,
  "currency": string or null,
  "line_items": [
    {
      "description": string or null,
      "quantity": number or null,
      "unit_price": number or null,
      "line_total": number or null
    }
  ],
  "subtotal": number or null,
  "tax": number or null,
  "grand_total": number or null,
  "notes": string or null
}

Rules:
- Return one object even if the image is blurry, angled, handwritten, or only
  partially readable.
- Use numbers for numeric fields, without currency symbols or thousands
  separators. Use decimal values where needed.
- If a line item has no printed quantity, use 1 only when that is clearly the
  normal implied quantity; otherwise use null.
- Do not invent a subtotal, tax, total, merchant, date, currency, or line item.
- Keep "notes" short and mention important uncertainty such as "tax unreadable"
  or "image is partially cropped". Use null if there is no useful note.
""".strip()

# Parses the JSON into a readable object for dictionary conversion
def parse_json(res):
    response = res.strip()

    try:
        result = json.loads(res)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecode()

    for index, character in enumerate(response):
        if character == "{":
            try:
                result, position = decoder.raw_decode(
                    response[index:]
                )

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON object was found in Gemini's response.")

# Extracts the data out of the image
def analyse_image(img):
    img_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type

    if not img_bytes:
        raise ValueError("The uploaded file is empty.")

    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("Please upload a valid image file.")

    img_part = types.Part.from_bytes(
        data=img_bytes,
        mime_type=mime_type
    )

    client = genai.Client(api_key=gemini_key)

    model = "gemini-3.5-flash-lite"

    response = client.models.generate_content(
        model=model,
        contents=[
            img_part,
            EXTRACTION_PROMPT
        ],
        config=types.GenerateContentConfig(
               response_mime_type="application/json"
        )
    )

    return parse_json(response.text)

# Converts the output from Gemini into rows
def convert_to_rows(response, filename):
    line_items = response.get("line_items", [])

    if not line_items:
        line_items = [{}]

    rows = list()
    for item in line_items:
        row = {
            "source_file": filename,
            "merchant_name": response.get("merchant_name"),
            "date": response.get("date"),
            "currency": response.get("currency"),
            "description": item.get("description"),
            "quantity": item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "line_total": item.get("line_total"),
            "subtotal": response.get("subtotal"),
            "tax": response.get("tax"),
            "grand_total": response.get("grand_total"),
            "notes": response.get("notes"),
        }

        rows.append(row)

    return rows
        

# Setting up the streamlit page
st.set_page_config(
    page_title="Receipt Extractor",
    page_icon="🧾"
)

st.title("🧾 Receipt and Invoice Extractor")
st.write("Upload an image of a receipt or invoice and extract its data.")

# Upload a picture of the receipt
uploaded_file = st.file_uploader(
    "Choose a receipt or invoice image",
    type=["jpg", "jpeg", "png", "webp"]
)

# If there is nothing at the moment, create a list for the table
if "all_rows" not in st.session_state:
    st.session_state.all_rows = []

# If there is an uploaded file, display a button to analyse image and add to table
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=400)

    if st.button("Analyse image"):
        try:
            description = analyse_image(uploaded_file)

            rows = convert_to_rows(
                description,
                uploaded_file.name
            )

            df = pd.DataFrame(rows)

            st.session_state.all_rows.extend(rows)
            st.write(description)
            st.success("Receipt extracted successfully.")
        except ValueError as error:
            st.warning(f"The image could not be read completely: {error}")

        except Exception as error:
            st.error(
                f"""The Gemini request failed. Check your internet connection, 
                API key, model name, or free-tier quota.
                Error: {error}"""
            )

# Collate all information into one table and download into CSV file
st.subheader("Accumulated spreadsheet data")

if st.session_state.all_rows:
    dataframe = pd.DataFrame(st.session_state.all_rows)

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True
    )

    csv_data = dataframe.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="receipt_extractions.csv",
        mime="text/csv"
    )

    if st.button("Clear accumulated data"):
        st.session_state.all_rows = []
        st.rerun()

else:
    st.info("No receipts have been extracted yet.")

# Loading the Gemini Model and API Key

if gemini_key:
    st.success("Gemini API key loaded successfully.")
else:
    st.error("Gemini API key was not found.")
