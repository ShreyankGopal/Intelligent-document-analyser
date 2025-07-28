# document_selector_mmr.py

import os
import re
import json
import joblib
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# ------------------------------- Bullet + Heuristic Utilities -------------------------------
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
import joblib
import json
import re
from sklearn.preprocessing import MinMaxScaler

# def is_heading_heuristic(feat_row):
#     text = feat_row['Section Text'].strip()
#     if not text:
#         return False
#     fs = feat_row['Font Size Normalised']
#     bold = feat_row['Is Bold']
#     cap_ratio = feat_row['Capitalization Ratio']
#     starts_numbered = feat_row['Starts with Numbering']
#     ends_colon = text.endswith(':')
#     ends_dot = text.endswith('.') or text.endswith('?') or text.endswith('!')
#     words = text.split()
#     wc = len(words)
#     y_gap = feat_row.get('Y Gap Scaled', 0)
#     if wc>25:
#         return False
#     # Heuristic rules
#     # if starts_numbered and wc < 12:
#     #     return True
#     if fs > 0.85:  # Top 15% font sizes
#         return True
#     if bold and (wc <= 15 or cap_ratio > 0.5 or ends_colon):
#         return True
#     if cap_ratio > 0.6 and wc <= 8:
#         return True
#     if text.istitle() and wc <= 8:
#         return True
#     # if ends_dot:
#     #     return False
#     # if y_gap > 0.1:  # New vertical spacing feature
#     #     return True
#     return False
def is_bullet_point(text):
    """Check if text is a bullet point that should be ignored."""
    text = text.strip()

    # Common bullet point patterns
    bullet_patterns = [
        r'^[•·▪▫▬►‣⁃]\s*',  # Unicode bullet symbols
        r'^\*\s+',           # Asterisk bullets
        r'^-\s+',            # Dash bullets
        r'^—\s+',            # Em dash bullets
        r'^–\s+',            # En dash bullets
        r'^\+\s+',           # Plus bullets
        r'^>\s+',            # Greater than bullets
        r'^»\s+',            # Right guillemet bullets
        r'^○\s+',            # Circle bullets
        r'^□\s+',            # Square bullets
        r'^▪\s+',            # Black square bullets
        r'^▫\s+',            # White square bullets
    ]

    # Check if text matches any bullet pattern
    for pattern in bullet_patterns:
        if re.match(pattern, text):
            return True

    # Check for numbered lists that are very short (likely bullets)
    if re.match(r'^\d+[\.\)]\s*$', text) or re.match(r'^[a-zA-Z][\.\)]\s*$', text):
        return True

    # Check for very short standalone symbols
    if len(text) <= 3 and re.match(r'^[^\w\s]+$', text):
        return True

    return False

def should_ignore_text(text):
    """Check if text should be completely ignored."""
    text = text.strip()

    # Ignore empty or very short text
    if len(text) < 2:
        return True

    # Ignore bullet points
    if is_bullet_point(text):
        return True

    # Ignore standalone numbers or letters (likely page numbers or references)
    if re.match(r'^\d+$', text) or re.match(r'^[a-zA-Z]$', text):
        return True

    # Ignore common PDF artifacts
    artifacts = ['©', '®', '™', '...', '…']
    if text in artifacts:
        return True

    return False

def clean_text(text):
    """Clean text by removing bullet point prefixes but keeping the content."""
    text = text.strip()

    # Remove bullet point prefixes but keep the rest of the text
    bullet_patterns = [
        r'^[•·▪▫▬►‣⁃]\s*',  # Unicode bullet symbols
        r'^\*\s+',           # Asterisk bullets
        r'^-\s+',            # Dash bullets
        r'^—\s+',            # Em dash bullets
        r'^–\s+',            # En dash bullets
        r'^\+\s+',           # Plus bullets
        r'^>\s+',            # Greater than bullets
        r'^»\s+',            # Right guillemet bullets
        r'^○\s+',            # Circle bullets
        r'^□\s+',            # Square bullets
        r'^▪\s+',            # Black square bullets
        r'^▫\s+',            # White square bullets
    ]

    for pattern in bullet_patterns:
        text = re.sub(pattern, '', text)

    return text.strip()

def extract_features(text, pdf_path, page_num, font_size, is_bold, is_italic, position_y, y_gap):
    text_length = len(text)
    upper_count = sum(1 for c in text if c.isupper())
    total_alpha = sum(1 for c in text if c.isalpha())
    capitalization_ratio = upper_count / total_alpha if total_alpha > 0 else 0
    starts_with_numbering = bool(re.match(r'^\d+(\.\d+)*(\.|\))\s', text))
    dot_match = re.match(r'^(\d+\.)+(\d+)', text)
    num_dots_in_prefix = dot_match.group(1).count('.') if dot_match else 0

    return {
        'PDF Path': str(pdf_path),
        'Page Number': page_num,
        'Section Text': text,
        'Font Size': font_size,
        'Is Bold': is_bold,
        'Is Italic': is_italic,
        'Text Length': text_length,
        'Capitalization Ratio': capitalization_ratio,
        'Starts with Numbering': starts_with_numbering,
        'Position Y': position_y,
        'Prefix Dot Count': num_dots_in_prefix,
        'Y Gap': y_gap
    }

def analyze_pdf_sections(pdf_path):
    sections_data = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")['blocks']

            prev_line_y = None
            prev_font_size = None
            prev_bold = None
            prev_italic = None
            current_lines = []

            for block in blocks:
                if block['type'] != 0:  # Skip non-text blocks
                    continue

                for line in block['lines']:
                    spans = [s for s in line['spans'] if s['text'].strip()]
                    if not spans:
                        continue

                    line_text = " ".join(span['text'].strip() for span in spans)

                    # Skip if this line should be ignored
                    if should_ignore_text(line_text):
                        continue

                    # Clean the text (remove bullet prefixes but keep content)
                    cleaned_text = clean_text(line_text)
                    if not cleaned_text or should_ignore_text(cleaned_text):
                        continue

                    first_span = spans[0]
                    font_size = first_span['size']
                    font_flags = first_span['flags']
                    is_bold = (font_flags & 16) > 0
                    is_italic = (font_flags & 2) > 0
                    y_position = first_span['bbox'][1]  # Top Y coordinate

                    # Compute Y gap
                    if prev_line_y is None:
                        y_gap = None
                    else:
                        y_gap = abs(y_position - prev_line_y)
                    prev_line_y = y_position

                    same_style = (
                        prev_font_size is not None and
                        abs(prev_font_size - font_size) < 0.5 and
                        is_bold == prev_bold and
                        is_italic == prev_italic
                    )

                    if same_style:
                        current_lines.append(cleaned_text)
                    else:
                        if current_lines:
                            full_text = " ".join(current_lines)
                            # Only add if the combined text is meaningful
                            if not should_ignore_text(full_text) and len(full_text.strip()) > 2:
                                feat = extract_features(
                                    full_text, pdf_path, page_num,
                                    prev_font_size, prev_bold, prev_italic, prev_line_y, prev_y_gap
                                )
                                sections_data.append(feat)

                        current_lines = [cleaned_text]
                        prev_font_size = font_size
                        prev_bold = is_bold
                        prev_italic = is_italic
                        prev_y_gap = y_gap

            # Process final group for this page
            if current_lines:
                full_text = " ".join(current_lines)
                if not should_ignore_text(full_text) and len(full_text.strip()) > 2:
                    feat = extract_features(
                        full_text, pdf_path, page_num,
                        prev_font_size, prev_bold, prev_italic, prev_line_y, prev_y_gap
                    )
                    sections_data.append(feat)

        doc.close()

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        import traceback
        traceback.print_exc()

    return pd.DataFrame(sections_data)

def preprocess_features(df):
    if df.empty:
        return df

    df['Is Bold'] = df['Is Bold'].astype(int)
    df['Is Italic'] = df['Is Italic'].astype(int)
    df['Starts with Numbering'] = df['Starts with Numbering'].astype(int)

    font_sizes = sorted(df['Font Size'].unique(), reverse=True)
    font_size_rank_map = {size: rank + 1 for rank, size in enumerate(font_sizes)}
    df['Font Size Rank'] = df['Font Size'].map(font_size_rank_map)

    df['Font Size Normalised'] = df['Font Size']
    columns_to_normalize = ['Font Size Normalised', 'Text Length', 'Capitalization Ratio', 'Position Y', 'Prefix Dot Count']

    if len(df) > 0:
        scaler = MinMaxScaler()
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    # Font Ratio
    if not df['Font Size'].empty:
        body_font_size = df['Font Size'].mode()[0]
        df['Font Ratio'] = df['Font Size'] / body_font_size
    else:
        df['Font Ratio'] = 1.0

    # Font Count + Uniqueness
    df['Font Size Count'] = df['Font Size'].map(df['Font Size'].value_counts())
    df['Is Unique Font Size'] = (df['Font Size Count'] == 1).astype(int)

    # Y Gap Scaled per PDF
    df['Y Gap'] = df['Y Gap'].fillna(2)
    df['Y Gap'] = pd.to_numeric(df['Y Gap'], errors='coerce').fillna(2)

    def scale_column_per_pdf(group):
        if len(group) > 1 and group.std() > 0:
            scaler = MinMaxScaler()
            return scaler.fit_transform(group.values.reshape(-1, 1)).flatten()
        else:
            return [0] * len(group)

    df['Y Gap Scaled'] = df.groupby('PDF Path')['Y Gap'].transform(scale_column_per_pdf)
    df['Font Size Count'] = df.groupby('PDF Path')['Font Size Count'].transform(scale_column_per_pdf)

    #display(df)
    return df

def build_json_from_predictions(df):
    outline = []

    for _, row in df[df['Label'] != 'None'].iterrows():
        outline.append({
            "level": row['Label'],
            "text": row['Section Text'],
            "page": int(row['Page Number'])
        })

    # Check if any row has Label == 'Title'
    title_rows = df[df['Label'] == 'Title']
    if not title_rows.empty:
        title_text = title_rows.iloc[0]['Section Text']
    else:
        title_text = outline[0]['text'] if outline else "Untitled Document"

    return {
        "title": title_text,
        "outline": outline
    }


def process_pdfs(pdf_path,model):
    print(f"📄 Processing {pdf_path.name}...")
    features = ['Font Ratio', 'Font Size Rank', 'Text Length', 'Capitalization Ratio',
            'Position Y',  'Is Bold', 'Is Italic',
            'Starts with Numbering', 'Font Size Count', 'Is Unique Font Size']
    df = analyze_pdf_sections(pdf_path)
    if df.empty:
        print(f"⚠️ Skipping {pdf_path.name} — no extractable text.")
        return

    df = preprocess_features(df)
    if df.empty:
        print(f"⚠️ Skipping {pdf_path.name} — preprocessing failed.")
        return

    
    #df['Is_Heading_H'] = df.apply(is_heading_heuristic, axis=1)
     #   df.loc[df['Is_Heading_H'], 'Is Bold'] = 1
    df['Label'] = "none"
    #heading_rows = df[df['Is_Heading_H']].copy()
    try:
        #heading_rows['Label'] = model.predict(heading_rows[features])
        # df['Label'] = model.predict(df[features])
        df.loc[df.index, 'Label'] = model.predict(df[features])
    except Exception as e:
        print(f"❌ Prediction failed for {pdf_path.name}: {e}")
        return
    
    # heading_rows = df[df['Is_Heading_H']]
    # display(heading_rows)
    # structured_json = build_json_from_predictions(heading_rows)

    # output_path = output_dir / f"{pdf_path.stem}.json"
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(structured_json, f, indent=2, ensure_ascii=False)

    # print(f"✅ Done: {output_path.name}")
    return df
def mmr(query_emb, sections, lambda_param=0.72, top_k=5):
    selected, remaining = [], list(range(len(sections)))
    sim_q = [util.cos_sim(query_emb, s['embedding']).item() for s in sections]
    sim_doc = [[util.cos_sim(sections[i]['embedding'], sections[j]['embedding']).item() for j in range(len(sections))] for i in range(len(sections))]
    while len(selected) < top_k and remaining:
        if not selected:
            idx = np.argmax([sim_q[i] for i in remaining])
            selected.append(remaining.pop(idx))
        else:
            mmr_scores = []
            for idx in remaining:
                max_sim = max(sim_doc[idx][j] for j in selected)
                score = lambda_param * sim_q[idx] - (1 - lambda_param) * max_sim
                mmr_scores.append(score)
            idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(idx))
    return selected, sim_q
# ------------------------------- Main Processing -------------------------------
def main(input_json_path, model_path, input_dir, output_path):
    with open(input_json_path) as f:
        input_data = json.load(f)

    model = joblib.load(model_path)
    embedder = SentenceTransformer("./cached_model")
    persona = input_data['persona']['role']
    job = input_data['job_to_be_done']['task']
    query_text = f"{job} {persona}"
    query_embedding = embedder.encode(query_text, normalize_embeddings=True)

    all_dfs, section_data = [], []
    for doc in input_data['documents']:
        pdf_path = Path(input_dir) / doc['filename']
        df = process_pdfs(pdf_path,model)
        if df.empty: continue
        all_dfs.append(df)

    final_df = pd.concat(all_dfs)
    section_labels = ['Title', 'H1', 'H2']
    for i, row in final_df.iterrows():
        if row['Label'] in section_labels:
            heading = row['Section Text']
            body = []
            for j in range(i + 1, len(final_df)):
                next_row = final_df.iloc[j]
                if next_row['Label'] in section_labels:
                    break
                body.append(next_row['Section Text'])
            full_text = heading + ". " + " ".join(body)
            section_data.append({
                'Document': row['PDF Path'],
                'Page': row['Page Number'],
                'heading': heading,
                'text': full_text,
                'embedding': embedder.encode(full_text, normalize_embeddings=True)
            })

    selected_indices, sim_scores = mmr(query_embedding, section_data, top_k=5)
    now = datetime.now().isoformat()
    output = {
        "metadata": {
            "input_documents": [doc['filename'] for doc in input_data['documents']],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": now
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    for rank, idx in enumerate(selected_indices, start=1):
        section = section_data[idx]
        output['extracted_sections'].append({
            'document': section['Document'],
            'section_title': section['heading'],
            'importance_rank': rank,
            'page_number': section['Page']
        })
        output['subsection_analysis'].append({
            'document': section['Document'],
            'refined_text': section['text'],
            'page_number': section['Page']
        })

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="/app/input/input.json")
    parser.add_argument("--model_path", type=str, default="/app/heading_classifier_with_font_count_norm_textNorm_5.pkl")
    parser.add_argument("--input_dir", type=str, default="/app/input")
    parser.add_argument("--output_json", type=str, default="/app/output/output.json")
    args = parser.parse_args()
    main(args.input_json, args.model_path, args.input_dir, args.output_json)
