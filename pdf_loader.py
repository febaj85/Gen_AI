from pypdf import PdfReader

def is_heading(line):
    return line.isupper() and len(line.split()) < 10

def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
        # print(chunks,"chunk")
    return chunks

def extract_sections(pdf_path):
    reader = PdfReader(pdf_path)
    sections = []

    current_heading = "INTRODUCTION"
    buffer = ""
    page_no = 0

    for page_no, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        for line in text.split("\n"):
            line = line.strip()
            if is_heading(line):
                if buffer.strip():
                    sections.append({
                        "heading": current_heading,
                        "text": buffer.strip(),
                        "page": page_no
                    })
                current_heading = line
                buffer = ""
            else:
                buffer += line + " "

    if buffer.strip():
        sections.append({
            "heading": current_heading,
            "text": buffer.strip(),
            "page": page_no
        })

    return sections
