# torch 환경 롤백 방법

GPU 사용을 위해 torch를 cu130 빌드에서 cu128 빌드로 교체했다.
(드라이버 570.211.01 = CUDA 12.8, torch 2.13.0+cu130 은 CUDA 13.0 요구 -> cuda unavailable)

변경 전: torch==2.13.0 (+cu130), nvidia-*-cu13 계열
변경 후: torch==2.11.0 (+cu128), nvidia-*-cu12 계열

원래대로 되돌리려면:

    .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
