import pdfplumber
import pandas as pd
import re
import json
import unicodedata
import os
import logging
from difflib import get_close_matches

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_ordered_toc(pdf_path, start_page=3, end_page=5, fuzzy_threshold=0.8):
    """
    Extracts an ordered Table of Contents (TOC) from the given PDF pages.
    Uses fuzzy matching to improve robustness.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(start_page - 1, end_page):  # zero-indexed
                if i < len(pdf.pages):
                    text = pdf.pages[i].extract_text()
                    if text:
                        full_text += text + "\n"
    except Exception as e:
        logging.error(f"Error reading TOC pages: {e}")
        return pd.DataFrame()

    lines = full_text.strip().split("\n")
    toc_entries = []
    for line in lines:
        match = re.match(r"(\d+)\s+(.+)", line.strip())
        if match:
            index = int(match.group(1))
            title = clean_text(match.group(2))
            toc_entries.append((index, title))

    # Deduplicate and keep strictly increasing index from 1
    ordered = []
    expected = 1
    for idx, title in toc_entries:
        if idx == expected:
            ordered.append({"index": idx, "title": title})
            expected += 1
    if not ordered:
        logging.warning("No TOC entries found with strict matching. Trying fuzzy matching...")
        # Fuzzy fallback: try to extract lines that look like TOC entries
        for line in lines:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 2 and parts[0].isdigit():
                idx = int(parts[0])
                title = clean_text(" ".join(parts[1:]))
                ordered.append({"index": idx, "title": title})
    if not ordered:
        logging.error("No TOC entries found even with fuzzy matching.")
    return pd.DataFrame(ordered)


def extract_pdf_body_text(pdf_path, start_page=6):
    """
    Extracts full text from the body of the PDF starting from the given page.
    """
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(start_page - 1, len(pdf.pages)):
                text = pdf.pages[i].extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        logging.error(f"Error extracting body text: {e}")
    return full_text


def split_text_by_titles(full_text, toc_df, max_sections=None, min_section_length=100):
    """
    Splits full text using the ordered TOC titles. Uses fuzzy matching if exact match fails.
    """
    sections = []
    titles = toc_df["title"].tolist()
    if max_sections:
        titles = titles[:max_sections]
    missing_titles = []
    for i in range(len(titles)):
        start_title = titles[i]
        end_title = titles[i + 1] if i + 1 < len(titles) else None
        # Try exact match first
        start_index = full_text.find(start_title)
        # Fuzzy fallback if not found
        if start_index == -1:
            candidates = get_close_matches(start_title, full_text.split("\n"), n=1, cutoff=0.7)
            if candidates:
                start_index = full_text.find(candidates[0])
        if start_index == -1:
            logging.warning(f"Title not found in text: '{start_title}'")
            missing_titles.append(start_title)
            continue
        end_index = full_text.find(end_title, start_index + len(start_title)) if end_title else len(full_text)
        section_text = clean_text(full_text[start_index:end_index])
        if len(section_text) < min_section_length:
            logging.info(f"Section too short, skipping: '{start_title}'")
            continue
        sections.append({
            "index": i + 1,
            "title": start_title,
            "text": section_text
        })
    if missing_titles:
        logging.warning(f"Missing {len(missing_titles)} titles in body text.")
    return sections


def split_large_sections(sections, max_words=500):
    new_sections = []
    for section in sections:
        words = section["text"].split()
        if len(words) <= max_words:
            new_sections.append(section)
        else:
            # Split into chunks of max_words
            for i in range(0, len(words), max_words):
                chunk_words = words[i:i+max_words]
                chunk_text = " ".join(chunk_words)
                new_section = {
                    "index": section["index"],
                    "title": section["title"] + (f" (part {i//max_words+1})" if len(words) > max_words else ""),
                    "text": chunk_text
                }
                new_sections.append(new_section)
    return new_sections


def save_sections_as_json(sections, output_path="book_sections.json"):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sections, f, indent=2, ensure_ascii=False)
        logging.info(f"✅ Saved {len(sections)} sections to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")


# ---------------- MAIN ------------------
if __name__ == "__main__":
    pdf_path = "MachineLearningYearning.pdf"  # Change if needed
    output_path = "book_sections.json"

    if not os.path.exists(pdf_path):
        logging.error(f"File not found: {pdf_path}")
        exit(1)

    toc_df = extract_ordered_toc(pdf_path, start_page=3, end_page=5)
    if toc_df.empty:
        logging.error("Failed to extract TOC.")
        exit(1)

    full_text = extract_pdf_body_text(pdf_path, start_page=6)
    if not full_text.strip():
        logging.error("No body text found.")
        exit(1)

    sections = split_text_by_titles(full_text, toc_df, max_sections=None)
    # Split large sections into max 500 words each
    sections = split_large_sections(sections, max_words=500)
    save_sections_as_json(sections, output_path)
    logging.info(f"Extraction complete. {len(sections)} sections saved.")
