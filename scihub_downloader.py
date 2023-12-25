import typing
import uuid
from pathlib import Path

from scidownl import scihub_download


def download(url: str, paper_type: typing.Literal['doi'] = 'doi') -> Path:
    assert paper_type == 'doi', "Now we support only doi paper type"
    output_file_name = f"{uuid.uuid4()}.pdf"
    scihub_download(
        url,
        paper_type=paper_type,
        out=output_file_name,
    )
    output_file_path = Path(output_file_name)
    if output_file_path.exists():
        return output_file_path
    raise FileNotFoundError("Can't download file")


if __name__ == '__main__':
    from pdf import read_file

    file_name = download("https://www.sciencedirect.com/science/article/pii/S1877042815009441?ref=pdf_download&fr=RR-2&rr=83b1120eefb83aad")
    print(read_file(Path(file_name)))
