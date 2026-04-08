"""
LLM Service: 透過 Ollama Python 套件連線語言模型，
根據月報指標自動生成模型表現分析摘要（80字以內，無 emoji）。
"""
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def generate_llm_summary(report, override_prompt=None):
    """
    根據 generate_monthly_report() 的結果，呼叫 LLM 生成摘要。

    Args:
        report: generate_monthly_report() 回傳的 dict。
        override_prompt: 如果提供，使用這個 prompt 取代 report['llm_prompt']。
                          用於雙窗格合併 prompt 的場景。

    Returns:
        str: LLM 生成的摘要文字，若失敗則回傳 fallback 文字。
    """
    # 檢查是否啟用 LLM
    if not getattr(settings, 'OLLAMA_ENABLED', False):
        logger.info("[LLM] Ollama is disabled in settings.")
        return _fallback_summary(report)

    prompt = override_prompt or report.get('llm_prompt', '')
    if not prompt:
        return _fallback_summary(report)

    ollama_host = getattr(settings, 'OLLAMA_HOST', 'http://192.168.63.184:11434')
    ollama_model = getattr(settings, 'OLLAMA_MODEL', 'gpt-oss:120b')

    try:
        import ollama
        client = ollama.Client(host=ollama_host)

        logger.info(f"[LLM] Calling {ollama_host}, model={ollama_model}")

        response = client.chat(
            model=ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3},  # 低溫度 → 更穩定的摘要
        )

        result = response['message']['content'].strip()

        # Post-process: remove markdown formatting
        import re
        result = result.replace('**', '')
        result = result.replace('##', '')
        result = result.replace('# ', '')
        result = re.sub(r'^---+\s*$', '', result, flags=re.MULTILINE)
        result = re.sub(r'^- ', '・', result, flags=re.MULTILINE)
        result = result.strip()

        # Limit length (dual prompt: 600 chars; single: 350)
        max_len = 600 if override_prompt else 350
        if len(result) > max_len:
            trimmed = result[:max_len]
            last_period = max(trimmed.rfind('。'), trimmed.rfind('\n'))
            if last_period > max_len * 0.6:
                result = trimmed[:last_period + 1]
            else:
                result = trimmed + '...'
        logger.info(f"[LLM] Response: {len(result)} chars")
        return result

    except ImportError:
        logger.warning("[LLM] ollama package not installed.")
        return _fallback_summary(report)

    except Exception as e:
        logger.error(f"[LLM] Ollama call failed: {e}")
        return _fallback_summary(report)


def _fallback_summary(report):
    """當 LLM 不可用時，產生結構化 fallback 摘要（包含雙窗格標記）。"""
    if not report.get('safety_gates'):
        return '資料不足，無法生成分析摘要。'

    failed = [g for g in report['safety_gates'] if not g['passed']]
    passed = [g for g in report['safety_gates'] if g['passed']]

    # Build aging status
    aging_indicators = report.get('aging_indicators', [])
    aging_lines = []
    for ai in aging_indicators:
        status_map = {'good': '正常', 'warning': '需注意', 'critical': '異常'}
        aging_lines.append(f"・{ai['label']}（{ai['metric']}）= {ai['value']}，狀態{status_map.get(ai['status'], '未知')}")

    # Use tagged format for both 90-day and 30-day (fallback uses same text for both)
    if not failed:
        status_text = (
            f"・所有安全閃門均達標（{len(passed)}/{len(passed)} 通過）\n"
            f"・整體預測品質維持在可接受範圍\n"
        )
        aging_text = ""
        if aging_lines:
            aging_text = "\n".join(aging_lines) + "\n"
        cause_text = "・模型與目前資料分布仍相容\n"
        action_text = "・建議維持定期監控頻率\n・可視資料量增長考慮重新訓練\n"
    else:
        failed_names = '、'.join(g['label'] for g in failed)
        status_text = (
            f"・{failed_names} 未達安全門檻\n"
            f"・{len(failed)}/{len(failed) + len(passed)} 項指標需要關注\n"
        )
        aging_text = ""
        if aging_lines:
            aging_text = "\n".join(aging_lines) + "\n"
        cause_text = "・資料分布可能已偏離訓練時的特徵\n・模型校準度可能已隨時間下降\n"
        action_text = "・檢視近期資料品質是否有變化\n・評估是否需要以最新資料重新訓練模型\n"

    summary = (
        f"[90天-現況] 模型本期表現\n{status_text}"
        f"\n[30天-現況] 模型本期表現\n{status_text}"
        f"\n[90天-老化] 老化指標檢視\n{aging_text}"
        f"\n[30天-老化] 老化指標檢視\n{aging_text}"
        f"\n[原因] 可能因素\n{cause_text}"
        f"\n[建議] 行動方案\n{action_text}"
    )
    return summary.strip()
