from pathlib import Path

from setuptools import find_packages, setup


def _read_requirements() -> list[str]:
    req = Path(__file__).resolve().parent / "requirements.txt"
    if not req.exists():
        return []
    lines = []
    for line in req.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


setup(
    name="vcpria",
    version="0.1.0",
    description="VC-pria: real-time hand sign recognition demo (OpenCV + MediaPipe + ML)",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=_read_requirements(),
)
