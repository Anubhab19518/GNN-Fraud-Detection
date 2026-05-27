#!/usr/bin/env python3
import json, base64, os, sys
from pathlib import Path

def save_image(b64str, out_path):
    try:
        data = base64.b64decode(b64str)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Failed to write {out_path}: {e}")
        return False


def extract(nb_path, out_dir):
    nb_path = Path(nb_path)
    out_dir = Path(out_dir)
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        return 1
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    written = []
    cell_idx = 0
    for cell in nb.get('cells', []):
        cell_idx += 1
        # attachments in markdown cells
        atts = cell.get('attachments', {})
        att_i = 0
        for name, att_dict in atts.items():
            for mime, fragments in att_dict.items():
                att_i += 1
                if isinstance(fragments, list):
                    b64 = ''.join(fragments)
                else:
                    b64 = fragments
                ext = 'png' if 'png' in mime else ('jpg' if 'jpeg' in mime or 'jpg' in mime else 'bin')
                out_path = out_dir / f'cell-{cell_idx:02d}-attachment-{att_i:02d}.{ext}'
                if save_image(b64, out_path):
                    written.append(out_path)
        # outputs in code cells
        out_list = cell.get('outputs', [])
        out_j = 0
        for output in out_list:
            out_j += 1
            data = output.get('data', {})
            for mime, b64 in data.items():
                if not (mime.startswith('image/') or mime in ('image/png','image/jpeg')):
                    continue
                if isinstance(b64, list):
                    b64str = ''.join(b64)
                else:
                    b64str = b64
                # Some outputs may include single-line small strings not base64; skip non-base64 safely
                try:
                    # Normalize b64 padding
                    b64str_clean = b64str.replace('\n', '').replace(' ', '')
                    ext = 'png' if 'png' in mime else ('jpg' if 'jpeg' in mime or 'jpg' in mime else 'bin')
                    out_path = out_dir / f'cell-{cell_idx:02d}-output-{out_j:02d}.{ext}'
                    if save_image(b64str_clean, out_path):
                        written.append(out_path)
                except Exception as e:
                    print(f"Skipping non-image output in cell {cell_idx} output {out_j}: {e}")
    print(f"Extracted {len(written)} images to {out_dir}")
    for p in written:
        print(p)
    return 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: nb_extract_images.py path/to/notebook.ipynb [out_dir]')
        sys.exit(2)
    nb = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'screenshots'
    sys.exit(extract(nb, out))
