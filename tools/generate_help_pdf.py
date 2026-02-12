from __future__ import annotations
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'docs' / 'help.md'
OUT = ROOT / 'help.pdf'

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT = 50
TOP = 800
FONT_SIZE = 11
LEADING = 15
WRAP_WIDTH = 42


def to_hex_utf16be(s: str) -> str:
    return s.encode('utf-16-be').hex().upper()


def prepare_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip('\n')
        if not line.strip():
            lines.append('')
            continue
        if line.startswith('```'):
            continue
        if len(line) <= WRAP_WIDTH:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=WRAP_WIDTH, break_long_words=True, break_on_hyphens=False))
    return lines


def make_page_stream(lines: list[str]) -> bytes:
    cmds = [
        'BT',
        f'/F1 {FONT_SIZE} Tf',
        f'{LEFT} {TOP} Td',
        f'{LEADING} TL',
    ]
    first = True
    for line in lines:
        hexs = to_hex_utf16be(line)
        if first:
            cmds.append(f'<{hexs}> Tj')
            first = False
        else:
            cmds.append('T*')
            cmds.append(f'<{hexs}> Tj')
    cmds.append('ET')
    return ('\n'.join(cmds) + '\n').encode('ascii')


def build_pdf(text: str) -> bytes:
    all_lines = prepare_lines(text)
    lines_per_page = int((TOP - 60) / LEADING)
    pages = [all_lines[i:i + lines_per_page] for i in range(0, len(all_lines), lines_per_page)]

    objects: list[bytes] = []

    # 1: catalog, 2: pages, 3: font
    objects.append(b'<< /Type /Catalog /Pages 2 0 R >>')
    objects.append(b'<< /Type /Pages /Kids [] /Count 0 >>')
    objects.append(b'<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H >>')

    page_ids = []
    for p_lines in pages:
        stream = make_page_stream(p_lines)
        content_obj = f'<< /Length {len(stream)} >>\nstream\n'.encode('ascii') + stream + b'endstream'
        objects.append(content_obj)
        content_id = len(objects)

        page_obj = (
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] '
            f'/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>'
        ).encode('ascii')
        objects.append(page_obj)
        page_ids.append(len(objects))

    kids = ' '.join(f'{pid} 0 R' for pid in page_ids)
    objects[1] = f'<< /Type /Pages /Kids [ {kids} ] /Count {len(page_ids)} >>'.encode('ascii')

    out = bytearray(b'%PDF-1.4\n%\xE2\xE3\xCF\xD3\n')
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f'{i} 0 obj\n'.encode('ascii'))
        out.extend(obj)
        out.extend(b'\nendobj\n')

    xref_start = len(out)
    out.extend(f'xref\n0 {len(objects)+1}\n'.encode('ascii'))
    out.extend(b'0000000000 65535 f \n')
    for off in offsets[1:]:
        out.extend(f'{off:010d} 00000 n \n'.encode('ascii'))
    out.extend(
        f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n'.encode('ascii')
    )
    return bytes(out)


def main() -> None:
    text = SRC.read_text(encoding='utf-8')
    pdf = build_pdf(text)
    OUT.write_bytes(pdf)
    print(f'Generated: {OUT}')


if __name__ == '__main__':
    main()
