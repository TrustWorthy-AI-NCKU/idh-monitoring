"""
LLM Service: 透過 Ollama Python 套件連線語言模型，
根據月報指標自動生成模型表現分析摘要（80字以內，無 emoji）。
"""
import logging
from django.conf import settings

logger = logging.getLogger(__name__)


def generate_llm_summary(report):
    """
    根據 generate_monthly_report() 的結果，呼叫 LLM 生成 80 字以內的摘要。

    Args:
        report: generate_monthly_report() 回傳的 dict。

    Returns:
        str: LLM 生成的摘要文字，若失敗則回傳 fallback 文字。
    """
    # 檢查是否啟用 LLM
    if not getattr(settings, 'OLLAMA_ENABLED', False):
        logger.info("[LLM] Ollama is disabled in settings.")
        return _fallback_summary(report)

    prompt = report.get('llm_prompt', '')
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

        # Limit length (prompt asks for ≤250 chars, add buffer)
        if len(result) > 350:
            # Cut at last complete sentence within limit
            trimmed = result[:350]
            last_period = max(trimmed.rfind('。'), trimmed.rfind('。'), trimmed.rfind('\n'))
            if last_period > 200:
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
    """當 LLM 不可用時，產生模板式 fallback 摘要。"""
    if not report.get('safety_gates'):
        return '資料不足，無法生成分析摘要。'

    failed = [g for g in report['safety_gates'] if not g['passed']]

    if not failed:
        return '本期所有安全閘門指標均達標，模型運作正常，建議持續定期監測。'

    failed_names = '、'.join(g['label'] for g in failed)
    return f'本期{failed_names}未達安全門檻，建議檢視近期資料品質並評估是否需要重新訓練模型。'
