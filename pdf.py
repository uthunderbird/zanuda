from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def read_file(file_path: Path) -> str | None:
    try:
        reader = PdfReader(file_path)
    except PdfReadError:
        return None
    text = ''
    for page in reader.pages:
        text += ' ' + page.extract_text(0, 90)
    return text


if __name__ == '__main__':
    print(read_file(Path('gvion2016.pdf')))
