# Run Multilingual-Refusal in Google Colab

Easiest ways to get the project running in Colab.

---

## Option A: Clone from GitHub (recommended)

If your project is on GitHub, Colab can clone it in one cell.

### 1. Push your repo to GitHub (if you haven’t)

```bash
git remote add origin https://github.com/YOUR_USERNAME/Multilingual-Refusal.git
git push -u origin main
```

### 2. In Colab – run these cells in order

**Cell 1: Clone repo and go to project**

```python
!git clone https://github.com/IvAnastasia/Multilingual-Refusal.git
%cd Multilingual-Refusal
```

Replace `IvAnastasia/Multilingual-Refusal` with your fork if different.

**Cell 2: Install dependencies**

```python
# Minimal deps (translation + WildGuard): fast
!pip install -r requirements-colab.txt -q

# Or full project deps (pipeline, models, etc.): slower
# !pip install -r requirements-macos.txt -q
```

**Cell 3: Add project to path and set API key**

```python
import sys
import os

# So "from dataset...", "from pipeline...", "from evaluators..." work
sys.path.insert(0, "/content/Multilingual-Refusal")

# Google Translate API key (for translate_data.py and multi_inference.py back-translation)
# Get key: https://console.cloud.google.com/apis/credentials
os.environ["GOOGLE_TRANSLATE_API_KEY"] = "YOUR_API_KEY_HERE"
```

**Cell 4: Run what you need**

```python
# Example: translate a few instructions to another language
# !python scripts/translate_data.py   # edit target_langs / data_type in script first

# Example: run WildGuard evaluator only (see evaluators/COLAB_WILDGUARD.md)
# from evaluators.wildguard import WildGuardEvaluator
# evaluator = WildGuardEvaluator()
```

---

## Option B: Upload from your computer (no GitHub)

### 1. Zip the project locally

From the parent of `Multilingual-Refusal`:

```bash
zip -r Multilingual-Refusal.zip Multilingual-Refusal -x "*.git*" -x "*__pycache__*" -x "*.pyc" -x "output/*" -x "pipeline/runs/*"
```

Or in Finder: right‑click `Multilingual-Refusal` → Compress, then delete `.git` and large `output/` / `pipeline/runs/` if you don’t need them.

### 2. In Colab – run these cells

**Cell 1: Upload zip and unzip**

```python
from google.colab import files
import zipfile
import os

uploaded = files.upload()  # pick Multilingual-Refusal.zip
zip_path = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall("/content")
%cd /content/Multilingual-Refusal
```

**Cell 2: Install dependencies**

```python
!pip install -r requirements-colab.txt -q
```

**Cell 3: Path + API key**

```python
import sys
import os
sys.path.insert(0, "/content/Multilingual-Refusal")
os.environ["GOOGLE_TRANSLATE_API_KEY"] = "YOUR_API_KEY_HERE"
```

**Cell 4: Run your script**

```python
!python scripts/translate_data.py
# or
!python scripts/multi_inference.py --config configs/cfg.yaml --type harmful
```

---

## Option C: Use Google Drive

### 1. Put project on Drive

Upload `Multilingual-Refusal.zip` (or the folder) to Google Drive, e.g. `My Drive/Multilingual-Refusal`.

### 2. In Colab – mount Drive and use project**

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/drive/MyDrive/Multilingual-Refusal   # adjust path if needed

import sys
sys.path.insert(0, "/content/drive/MyDrive/Multilingual-Refusal")

!pip install -r requirements-colab.txt -q
os.environ["GOOGLE_TRANSLATE_API_KEY"] = "YOUR_API_KEY_HERE"
```

Then run scripts as in Option B.

---

## requirements-colab.txt

Use **requirements-colab.txt** (in the repo) for a smaller, Colab-friendly install (translation, evaluators, basic pipeline). For the full pipeline with all dependencies, use **requirements-macos.txt** (no CUDA-only packages).

---

## Quick reference

| Goal                    | After setup (path + deps + API key) |
|-------------------------|--------------------------------------|
| Translate instructions  | `!python scripts/translate_data.py`  |
| Run inference + back-tr | `!python scripts/multi_inference.py --config configs/cfg.yaml --type harmful` |
| Run WildGuard only      | See **evaluators/COLAB_WILDGUARD.md** |

**API key:** [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → Create credentials → API key → enable Cloud Translation API.
