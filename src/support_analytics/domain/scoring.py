from __future__ import annotations

from support_analytics.domain.models import ConversationArtifact, RuleSignals

CUSTOMER_GRATITUDE_MARKERS = (
    "дякую",
    "спасибі",
    "добре, дякую",
    "окей, дякую",
)
CUSTOMER_FRUSTRATION_MARKERS = (
    "це вже вдруге",
    "це неприйнятно",
    "я розчарований",
    "я незадоволений",
    "нічого не змінилося",
    "це не працює",
)
UNRESOLVED_MARKERS = (
    "не працює",
    "не допомогло",
    "нічого не змінилося",
    "я все ще не можу",
    "проблема залишилась",
    "спробую ще раз",
)
AGENT_APOLOGY_MARKERS = ("перепрошую", "вибачте", "шкода, що")
AGENT_CERTAINTY_RISK_MARKERS = (
    "це неможливо",
    "це точно не наша сторона",
    "у нас таких проблем немає",
    "вам треба просто почекати",
)


def analyze_rule_signals(conversation: ConversationArtifact) -> RuleSignals:
    signals = RuleSignals()
    for turn in conversation.turns:
        text = turn.message.lower()
        if turn.role == "customer":
            if any(marker in text for marker in CUSTOMER_GRATITUDE_MARKERS):
                signals.customer_gratitude_count += 1
            if any(marker in text for marker in CUSTOMER_FRUSTRATION_MARKERS):
                signals.customer_frustration_count += 1
            if any(marker in text for marker in UNRESOLVED_MARKERS):
                signals.unresolved_customer_signal_count += 1
        else:
            if any(marker in text for marker in AGENT_APOLOGY_MARKERS):
                signals.agent_apology_count += 1
            if any(marker in text for marker in AGENT_CERTAINTY_RISK_MARKERS):
                signals.agent_certainty_risk_count += 1

    signals.hidden_dissatisfaction_hint = (
        signals.customer_gratitude_count > 0 and signals.unresolved_customer_signal_count > 0
    )
    signals.explicit_conflict_hint = signals.customer_frustration_count > 0
    return signals
