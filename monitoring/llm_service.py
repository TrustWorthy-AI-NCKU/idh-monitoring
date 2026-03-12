"""
LLM Service: 透過 Ollama Python 套件連線遠端語言模型，
根據監控指標自動生成模型表現分析報告。
"""
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def _build_analysis_prompt(alerts, analysis_info):
    """
    將 generate_alerts() 產出的結構化告警資料，
    組裝成語言模型能理解的中文 prompt。
    """
    # 篩選出真正的異常告警（排除 info 級別）
    abnormal_alerts = [a for a in alerts if a.get('level') in ('critical', 'warning')]

    if not abnormal_alerts:
        return None  # 沒有異常，不需要呼叫 LLM

    # 組裝指標摘要
    metrics_lines = []
    for alert in abnormal_alerts:
        level_label = '🚨嚴重' if alert['level'] == 'critical' else '⚠️警告'
        metric = alert.get('metric', '未知')
        value = alert.get('value')
        value_str = f'{value:.4f}' if isinstance(value, (int, float)) else str(value)
        category = alert.get('category', '')
        metrics_lines.append(
            f"- [{level_label}] {category} | {metric} = {value_str}"
        )

    metrics_summary = '\n'.join(metrics_lines)

    # 組裝詳細描述
    detail_lines = []
    for alert in abnormal_alerts:
        detail_lines.append(f"【{alert.get('title', '')}】\n{alert.get('detail', '')}")

    detail_text = '\n\n'.join(detail_lines)

    # 時間範圍
    time_range = ''
    if analysis_info.get('start_date') and analysis_info.get('end_date'):
        time_range = (
            f"分析時間範圍：{analysis_info['start_date']} ~ {analysis_info['end_date']}，"
            f"共 {analysis_info.get('total_windows', '?')} 個時間窗口，"
            f"趨勢分析取最近 {analysis_info.get('trend_windows', '?')} 個窗口。"
        )

    prompt = f"""你是一位資深的機器學習工程師與臨床資料科學家，專精於醫療預測模型的監控與維護。
以下是一個用於預測「血液透析中低血壓 (IDH)」的機器學習模型的監控報告。

{time_range}

## 異常指標摘要
{metrics_summary}

## 詳細告警內容
{detail_text}

## 請你根據以上資訊，用繁體中文進行以下分析：

1. **根本原因推測**：根據這些指標的異常組合，推測最可能導致模型表現變差的根本原因（例如：資料族群改變、季節性因素、資料收集流程變更、模型老化等）。
2. **各指標關聯分析**：說明這些異常指標之間的因果關係（例如：資料偏移是否導致了效能下降？校準度失準是否與特定特徵漂移有關？）。
3. **建議行動方案**：提出具體的改善建議，按優先順序排列。

請用條列式回答，保持簡潔專業。請用70字以內回答。"""

    return prompt


def generate_llm_analysis(alerts, analysis_info):
    """
    呼叫遠端 Ollama 語言模型，生成模型表現分析報告。

    Args:
        alerts: generate_alerts() 產出的告警列表。
        analysis_info: 分析時間範圍等資訊。

    Returns:
        str: LLM 生成的分析文字，若失敗或無需分析則回傳 None。
    """
    # 檢查是否啟用 LLM
    if not getattr(settings, 'OLLAMA_ENABLED', False):
        logger.info("[LLM] Ollama is disabled in settings.")
        return None

    # 組裝 prompt
    prompt = _build_analysis_prompt(alerts, analysis_info)
    if prompt is None:
        logger.info("[LLM] No abnormal alerts, skipping LLM call.")
        return None

    # 呼叫 Ollama
    ollama_host = getattr(settings, 'OLLAMA_HOST', 'http://192.168.63.184:11434')
    ollama_model = getattr(settings, 'OLLAMA_MODEL', 'gpt-oss:120b')

    try:
        import ollama
        client = ollama.Client(host=ollama_host)

        logger.info(f"[LLM] Calling Ollama at {ollama_host}, model={ollama_model}")

        response = client.chat(
            model=ollama_model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            options={
                'temperature': 0.7,
            },
        )

        result = response['message']['content']
        logger.info(f"[LLM] Response received, length={len(result)} chars.")
        return result

    except ImportError:
        logger.warning("[LLM] ollama package not installed. Run: pip install ollama")
        return "⚠️ 尚未安裝 ollama 套件。請執行 pip install ollama"

    except Exception as e:
        logger.error(f"[LLM] Ollama call failed: {e}")
        return f"⚠️ AI 分析暫時無法使用（{str(e)}）"
