from __future__ import annotations

import json

from support_analytics.domain.models import ConversationArtifact, ScenarioBlueprint

GENERATOR_SYSTEM_PROMPT = """
You generate realistic customer support chats for QA analytics datasets.
Always follow the blueprint exactly.
Keep the dialogue natural, concise, and consistent with the requested language.
Do not add markdown, timestamps, or narration outside the schema.
Preserve subtle dissatisfaction when the blueprint requires hidden dissatisfaction.
If agent errors are required, make them realistic rather than cartoonish.
""".strip()

EVALUATOR_SYSTEM_PROMPT = """
You are a strict QA auditor for customer support conversations.
Classify the actual customer intent, customer satisfaction, agent quality, and agent mistakes.
Use only the labels allowed by the schema.
Return only the structured JSON response that matches the schema.
""".strip()


def build_generation_prompt(blueprint: ScenarioBlueprint) -> str:
    blueprint_json = json.dumps(blueprint.model_dump(mode="json"), ensure_ascii=False, indent=2)
    return f"""
Generate one customer support conversation using this blueprint.

Requirements:
- Language: {blueprint.language}
- Use {blueprint.min_turns} to {blueprint.max_turns} turns.
- Alternate roles naturally between customer and agent.
- The transcript must feel like a real live chat, not a script.
- Keep each message focused and realistic for a SaaS support team.
- The topic, resolution, tone, and customer satisfaction must match the blueprint.
- If hidden dissatisfaction is true, the customer should be formally polite near the end while the problem remains unresolved or only partially addressed.
- Reflect all required story beats.
- Reflect all required agent errors, when present.

Blueprint:
{blueprint_json}
""".strip()


def build_evaluation_prompt(conversation: ConversationArtifact) -> str:
    turns = [
        {
            "turn_index": turn.turn_index,
            "role": turn.role,
            "message": turn.message,
        }
        for turn in conversation.turns
    ]
    conversation_json = json.dumps(
        {
            "conversation_id": conversation.conversation_id,
            "language": conversation.language,
            "support_channel": conversation.support_channel,
            "title": conversation.title,
            "scenario_summary": conversation.scenario_summary,
            "turns": turns,
            "final_customer_signal": conversation.final_customer_signal,
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""
Evaluate the following support conversation.

Required output:
- intent: one of payment_issue, technical_error, account_access, plan_question, refund_request, other
- satisfaction: one of satisfied, neutral, unsatisfied
- quality_score: integer from 1 to 5 where 5 is excellent and 1 is very poor
- agent_mistakes: zero or more of ignored_question, incorrect_info, rude_tone, no_resolution, unnecessary_escalation

Judging rules:
- Choose other only when the conversation does not fit the supported intent categories.
- Choose satisfied only when the customer leaves with a clearly positive or resolved outcome.
- Choose unsatisfied when the customer remains blocked, frustrated, or the issue is unresolved.
- Use no_resolution when the agent fails to bring the conversation to a usable resolution.
- Use ignored_question when the agent leaves a direct customer question unanswered.
- Use incorrect_info when the agent gives misleading, wrong, or overconfident information.
- Use rude_tone when the agent is dismissive, sharp, or unnecessarily defensive.
- Use unnecessary_escalation when escalation is avoidable and the agent escalates anyway.

Conversation:
{conversation_json}
""".strip()
