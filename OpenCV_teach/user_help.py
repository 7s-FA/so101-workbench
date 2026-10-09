"""Open the illustrated, offline user manual from either application window."""
from pathlib import Path
import webbrowser


def open_manual():
    manual = Path(__file__).resolve().parent / 'guide' / '사용설명서.html'
    if not manual.is_file():
        raise FileNotFoundError(manual)
    if not webbrowser.open(manual.as_uri()):
        raise RuntimeError(f'브라우저에서 이 파일을 여세요: {manual}')
