# Troubleshooting

## `setup.bat`: "Python 3.9-3.13 was not found"
PaddlePaddle 3.2 has no wheels for Python 3.14. Install Python 3.12 from python.org (it can live next to other versions) and run `setup.bat` again. The script tries the `py` launcher first, then `python` / `python3` on PATH, and prints what it found.

## `setup.bat`: environment "moved due to redirects" / path not found
Microsoft Store Python redirects `%LOCALAPPDATA%` into a private package folder, so tools cannot find the environment there. The environment therefore lives in `%USERPROFILE%\.pdf-ocr\venv`, outside AppData.

## The console says `device: cpu`, but the PC has an NVIDIA GPU
The CPU build of PaddlePaddle is installed. Run `setup.bat` again (it installs the CUDA 12.9 GPU build when `nvidia-smi` is found). Check with `"%USERPROFILE%\.pdf-ocr\venv\Scripts\python.exe" -c "import paddle; print(paddle.__version__, paddle.device.is_compiled_with_cuda())"`. RTX 50-series (Blackwell) needs a CUDA 12.9 build and a driver that supports CUDA 12.9 or higher.

## `--structure` says it needs extra packages
Run `setup.bat` again, or `"%USERPROFILE%\.pdf-ocr\venv\Scripts\python.exe" -m pip install "paddlex[ocr]==3.7.2"`. The first `--structure` run also downloads the layout and table models (a few hundred MB) into `%USERPROFILE%\.paddlex`.

## The right-click menu does nothing / is missing
If clicking "OCR (PaddleOCR)" itself gives "This file does not have an app associated with it", the menu was installed by an older version; run `install_context_menu.bat` again. Windows 11 shows it under "Show more options". Run `install_context_menu.bat` again after moving the folder. A multi-selection opens ONE console after about 3 seconds; that is the queue collecting the files.

## GPU error during OCR
The tool falls back to the CPU on its own and prints a warning. To force the CPU: `--device cpu`.

## Runtime error about oneDNN / MKLDNN on CPU
The tool retries once without MKLDNN. To start without it: `--no-mkldnn`.

## Models cannot be downloaded
Put the model folders into `models\` next to `pdf_ocr.py` (see README, "Manual / offline models"). For Russian: `PP-OCRv5_server_det` and `eslav_PP-OCRv5_mobile_rec`.

## Words are glued together, amounts look like `86 о00`
Known recognition errors on some scans. Common cases are corrected by the cleanup (`--no-fix` disables it); glued words are not. Try `--dpi 250` or `300`, or compare with the `cyrillic_PP-OCRv5_mobile_rec` model via `--rec-model-dir`.

## A page is "low confidence"
Stamps, handwriting, photos taken at an angle. Check that page manually; `--auto-rotate` helps with sideways or upside-down pages.

## No searchable PDF is produced
No Unicode font was found. Pass `--font C:\Windows\Fonts\arial.ttf` (or any TTF with Cyrillic). Pages with a `/Rotate` setting get no text layer.
