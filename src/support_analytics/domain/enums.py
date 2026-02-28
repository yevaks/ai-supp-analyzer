from __future__ import annotations

from enum import StrEnum


class IssueType(StrEnum):
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_ERROR = "technical_error"
    ACCOUNT_ACCESS = "account_access"
    PLAN_QUESTION = "plan_question"
    REFUND_REQUEST = "refund_request"


class CaseClass(StrEnum):
    SUCCESSFUL = "successful"
    PROBLEMATIC = "problematic"
    CONFLICT = "conflict"
    AGENT_MISTAKE = "agent_mistake"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    UNRESOLVED = "unresolved"


class CustomerSatisfaction(StrEnum):
    VERY_DISSATISFIED = "very_dissatisfied"
    DISSATISFIED = "dissatisfied"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    DELIGHTED = "delighted"


class ActorRole(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"


class AgentErrorType(StrEnum):
    TONE = "tone"
    LOGIC = "logic"
    POLICY = "policy"
    ACCURACY = "accuracy"
    PROCESS = "process"
    EMPATHY = "empathy"


class SupportIntent(StrEnum):
    PAYMENT_ISSUE = "payment_issue"
    TECHNICAL_ERROR = "technical_error"
    ACCOUNT_ACCESS = "account_access"
    PLAN_QUESTION = "plan_question"
    REFUND_REQUEST = "refund_request"
    OTHER = "other"


class SupportSatisfaction(StrEnum):
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    UNSATISFIED = "unsatisfied"


class AgentMistakeTag(StrEnum):
    IGNORED_QUESTION = "ignored_question"
    INCORRECT_INFO = "incorrect_info"
    RUDE_TONE = "rude_tone"
    NO_RESOLUTION = "no_resolution"
    UNNECESSARY_ESCALATION = "unnecessary_escalation"
