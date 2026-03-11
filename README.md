# IDH 監測儀表板 (IDH Monitoring Dashboard)

基於 **Django + Plotly Dash** 的透析病人 IDH（血液透析中低血壓）風險模型監測介面。支援滑動時間窗口分析、特徵漂移偵測（JS Divergence / PSI）、模型效能追蹤（AUROC / AUPRC / F1）及自動告警。

---

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/YOUR_USERNAME/idh-monitoring.git
cd idh-monitoring
```

### 2. 建立 Python 虛擬環境（建議 Python 3.9+）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 4. 設定環境變數

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

接著開啟 `.env`，將 `SECRET_KEY` 換成一組新的密鑰（可用以下指令產生）：

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

### 5. 執行資料庫 migration

```bash
python manage.py migrate
```

### 6. 啟動伺服器

```bash
python manage.py runserver
```

開啟瀏覽器前往 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 使用方式

啟動後，在監測介面中：

1. **上傳資料檔**（`.csv`）：包含病人透析紀錄，需有 `Session_Date` 欄位。
2. **上傳模型檔**（`.pkl` / `.joblib`）：已訓練的分類模型。
3. （選用）**上傳 Baseline 訓練資料**：用於計算漂移程度的基準。
4. 點選 **分析** 即可產生滑動窗口效能圖與特徵漂移熱圖。

> ⚠️ **注意**：資料與模型不包含在此 repository 中，需自行提供。

---

## 系統需求

- Python 3.9+
- 詳見 `requirements.txt`

---

## 專案結構

```
idh_monitoring/
├── manage.py
├── requirements.txt
├── .env.example          ← 環境變數範本
├── idh_monitoring/       ← Django 主設定
│   ├── settings.py
│   └── urls.py
└── monitoring/           ← 監測應用程式
    ├── services.py       ← 核心計算邏輯（滑窗、漂移、告警）
    ├── dash_app.py       ← Plotly Dash 互動介面
    ├── views.py
    └── templates/
```
