def build_generation_prompt(n: int) -> str:
    return f"""
Ти — генератор синтетичних даних для оцінки LLM у задачах customer support.

Згенеруй рівно {n} чатів українською мовою.

Поверни тільки JSON такого формату:

{{
  "dataset": [
    {{
      "chat_id": "...",
      "scenario_tags": [...],
      "turns": [
        {{"role": "customer", "text": "..."}},
        {{"role": "agent", "text": "..."}}
      ]
    }}
  ]
}}

Без markdown. Тільки валідний JSON.
"""
    

def build_analysis_prompt(dialog_text: str) -> str:
    return f"""
Проаналізуй діалог служби підтримки.

Поверни тільки JSON:

{{
  "intent": "...",
  "satisfaction": "satisfied|neutral|unsatisfied",
  "quality_score": 1-5,
  "agent_mistakes": []
}}

Діалог:
{dialog_text}
"""