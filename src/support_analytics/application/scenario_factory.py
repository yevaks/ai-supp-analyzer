from __future__ import annotations

from itertools import product
from random import Random

from support_analytics.domain.enums import (
    AgentErrorType,
    CaseClass,
    CustomerSatisfaction,
    IssueType,
    ResolutionStatus,
)
from support_analytics.domain.models import AgentError, ScenarioBlueprint

SUPPORT_CHANNELS = ("live_chat", "web_widget", "billing_portal", "mobile_app_chat")
CUSTOMER_PROFILES = (
    "Малий бізнес-користувач, який самостійно адмініструє акаунт.",
    "Операційний менеджер середньої команди з високою залежністю від сервісу.",
    "Фаундер стартапу, який швидко реагує на збої та витрати.",
    "Фінансовий менеджер, що контролює оплати та повернення.",
    "Спеціаліст підтримки клієнтів, який ескалує питання від імені своєї команди.",
)

ISSUE_LIBRARY: dict[IssueType, list[dict[str, object]]] = {
    IssueType.PAYMENT_ISSUE: [
        {
            "prompt": "Подвійне списання після повторної спроби оплатити підписку.",
            "beats": [
                "клієнт бачить два списання за один період",
                "агент просить або перевіряє ідентифікатор платежу",
                "обговорюється різниця між холдом банку та реальним списанням",
            ],
        },
        {
            "prompt": "Платіж завис у статусі processing, але доступ до сервісу не активувався.",
            "beats": [
                "клієнт повідомляє про списання без активації тарифу",
                "агент перевіряє статус транзакції",
                "у діалозі згадується вплив на доступ до функцій",
            ],
        },
    ],
    IssueType.TECHNICAL_ERROR: [
        {
            "prompt": "Після оновлення застосунку не працює експорт звітів.",
            "beats": [
                "клієнт описує відтворювану технічну помилку",
                "агент уточнює середовище або кроки відтворення",
                "у чаті згадуються логічні або технічні кроки діагностики",
            ],
        },
        {
            "prompt": "Система повертає помилку 500 при спробі зберегти налаштування.",
            "beats": [
                "клієнт описує конкретний код або симптом помилки",
                "агент збирає контекст і пропонує перевірку",
                "обговорюється тимчасовий обхідний шлях або ескалація",
            ],
        },
    ],
    IssueType.ACCOUNT_ACCESS: [
        {
            "prompt": "Клієнт не може увійти після зміни пароля та не отримує лист відновлення.",
            "beats": [
                "клієнт не може відновити доступ до акаунту",
                "агент перевіряє email, SSO або статус блокування",
                "у діалозі згадуються кроки безпеки або перевірки особи",
            ],
        },
        {
            "prompt": "Акаунт позначений як заблокований після кількох невдалих спроб входу.",
            "beats": [
                "клієнт повідомляє про блокування акаунту",
                "агент пояснює причину або процедуру розблокування",
                "у чаті згадується час очікування чи ручна перевірка",
            ],
        },
    ],
    IssueType.PLAN_QUESTION: [
        {
            "prompt": "Клієнт хоче зрозуміти різницю між Pro та Business тарифами.",
            "beats": [
                "клієнт ставить питання про можливості тарифів",
                "агент пояснює ліміти або функції планів",
                "обговорюється вплив зміни тарифу на поточні дані чи оплату",
            ],
        },
        {
            "prompt": "Клієнт сумнівається, чи варто переходити на річну оплату.",
            "beats": [
                "клієнт питає про умови тарифу",
                "агент пояснює умови білінгу або знижки",
                "в діалозі згадується момент зміни плану",
            ],
        },
    ],
    IssueType.REFUND_REQUEST: [
        {
            "prompt": "Клієнт просить повернення коштів після небажаного продовження підписки.",
            "beats": [
                "клієнт просить повернути кошти",
                "агент перевіряє дату списання та політику refund",
                "у діалозі обговорюється часткове або повне повернення",
            ],
        },
        {
            "prompt": "Клієнт оплатив не той тариф і хоче скасувати покупку.",
            "beats": [
                "клієнт пояснює помилкову покупку",
                "агент перевіряє статус активації тарифу",
                "у чаті згадується політика скасування та строки",
            ],
        },
    ],
}


class ScenarioFactory:
    def build(self, *, count: int, seed: int, language: str) -> list[ScenarioBlueprint]:
        issue_case_pairs = list(product(IssueType, CaseClass))
        if count < len(issue_case_pairs):
            raise ValueError(
                f"count must be at least {len(issue_case_pairs)} to cover all issue and case classes."
            )

        rng = Random(seed)
        planned_pairs = issue_case_pairs + [
            issue_case_pairs[index % len(issue_case_pairs)]
            for index in range(count - len(issue_case_pairs))
        ]
        rng.shuffle(planned_pairs)

        blueprints: list[ScenarioBlueprint] = []
        for index, (issue_type, case_class) in enumerate(planned_pairs):
            variant_pool = ISSUE_LIBRARY[issue_type]
            variant = variant_pool[(seed + index) % len(variant_pool)]
            case_targets = self._build_case_targets(
                issue_type=issue_type,
                case_class=case_class,
                rng=rng,
                index=index,
            )
            blueprints.append(
                ScenarioBlueprint(
                    conversation_id=f"chat_{seed:04d}_{index + 1:03d}",
                    language=language,
                    support_channel=SUPPORT_CHANNELS[(seed + index) % len(SUPPORT_CHANNELS)],
                    issue_type=issue_type,
                    case_class=case_class,
                    customer_profile=CUSTOMER_PROFILES[(seed * 3 + index) % len(CUSTOMER_PROFILES)],
                    scenario_prompt=str(variant["prompt"]),
                    customer_tone=case_targets["customer_tone"],
                    resolution_status=case_targets["resolution_status"],
                    visible_customer_satisfaction=case_targets["visible_customer_satisfaction"],
                    actual_customer_satisfaction=case_targets["actual_customer_satisfaction"],
                    hidden_dissatisfaction=case_targets["hidden_dissatisfaction"],
                    problem_solved=case_targets["problem_solved"],
                    target_answer_quality_score=case_targets["target_answer_quality_score"],
                    required_story_beats=self._build_story_beats(
                        base_beats=list(variant["beats"]),
                        case_class=case_class,
                        hidden_dissatisfaction=case_targets["hidden_dissatisfaction"],
                        problem_solved=case_targets["problem_solved"],
                    ),
                    required_agent_errors=case_targets["required_agent_errors"],
                    min_turns=6,
                    max_turns=10,
                    generation_seed=seed + index,
                )
            )
        return blueprints

    def _build_story_beats(
        self,
        *,
        base_beats: list[str],
        case_class: CaseClass,
        hidden_dissatisfaction: bool,
        problem_solved: bool,
    ) -> list[str]:
        beats = list(base_beats)
        if hidden_dissatisfaction:
            beats.append("наприкінці клієнт звучить ввічливо, але проблема лишається не до кінця вирішеною")
        if case_class == CaseClass.CONFLICT:
            beats.append("у середині діалогу напруга між клієнтом та агентом відчутно зростає")
        elif case_class == CaseClass.SUCCESSFUL:
            beats.append("агент приводить діалог до чіткого та перевіреного результату")
        elif not problem_solved:
            beats.append("кінцівка діалогу залишає ризик повторного звернення")
        return beats

    def _build_case_targets(
        self,
        *,
        issue_type: IssueType,
        case_class: CaseClass,
        rng: Random,
        index: int,
    ) -> dict[str, object]:
        if case_class == CaseClass.SUCCESSFUL:
            return {
                "customer_tone": "стримано-прагматичний",
                "resolution_status": ResolutionStatus.RESOLVED,
                "visible_customer_satisfaction": CustomerSatisfaction.SATISFIED
                if index % 2
                else CustomerSatisfaction.DELIGHTED,
                "actual_customer_satisfaction": CustomerSatisfaction.SATISFIED
                if index % 2
                else CustomerSatisfaction.DELIGHTED,
                "hidden_dissatisfaction": False,
                "problem_solved": True,
                "target_answer_quality_score": 92 - (index % 8),
                "required_agent_errors": [],
            }

        if case_class == CaseClass.PROBLEMATIC:
            hidden = index % 2 == 0
            return {
                "customer_tone": "стривожений",
                "resolution_status": ResolutionStatus.PARTIALLY_RESOLVED
                if hidden
                else ResolutionStatus.UNRESOLVED,
                "visible_customer_satisfaction": CustomerSatisfaction.SATISFIED
                if hidden
                else CustomerSatisfaction.NEUTRAL,
                "actual_customer_satisfaction": CustomerSatisfaction.DISSATISFIED
                if hidden
                else CustomerSatisfaction.NEUTRAL,
                "hidden_dissatisfaction": hidden,
                "problem_solved": False,
                "target_answer_quality_score": 62 - (index % 9),
                "required_agent_errors": self._build_agent_errors(
                    issue_type=issue_type,
                    case_class=case_class,
                    rng=rng,
                ),
            }

        if case_class == CaseClass.CONFLICT:
            return {
                "customer_tone": "роздратований",
                "resolution_status": ResolutionStatus.UNRESOLVED
                if index % 3
                else ResolutionStatus.PARTIALLY_RESOLVED,
                "visible_customer_satisfaction": CustomerSatisfaction.DISSATISFIED,
                "actual_customer_satisfaction": CustomerSatisfaction.VERY_DISSATISFIED,
                "hidden_dissatisfaction": False,
                "problem_solved": False,
                "target_answer_quality_score": 38 - (index % 7),
                "required_agent_errors": self._build_agent_errors(
                    issue_type=issue_type,
                    case_class=case_class,
                    rng=rng,
                ),
            }

        hidden = index % 3 == 0
        return {
            "customer_tone": "напружено-ввічливий" if hidden else "стресовий",
            "resolution_status": ResolutionStatus.PARTIALLY_RESOLVED
            if hidden
            else ResolutionStatus.UNRESOLVED,
            "visible_customer_satisfaction": CustomerSatisfaction.NEUTRAL
            if hidden
            else CustomerSatisfaction.DISSATISFIED,
            "actual_customer_satisfaction": CustomerSatisfaction.DISSATISFIED,
            "hidden_dissatisfaction": hidden,
            "problem_solved": False,
            "target_answer_quality_score": 47 - (index % 11),
            "required_agent_errors": self._build_agent_errors(
                issue_type=issue_type,
                case_class=case_class,
                rng=rng,
            ),
        }

    def _build_agent_errors(
        self,
        *,
        issue_type: IssueType,
        case_class: CaseClass,
        rng: Random,
    ) -> list[AgentError]:
        if case_class == CaseClass.SUCCESSFUL:
            return []

        issue_specific = {
            IssueType.PAYMENT_ISSUE: "агент некоректно трактує статус платежу або не перевіряє важливі деталі транзакції",
            IssueType.TECHNICAL_ERROR: "агент робить поспішний висновок без достатньої діагностики",
            IssueType.ACCOUNT_ACCESS: "агент пропускає важливий крок верифікації або відновлення доступу",
            IssueType.PLAN_QUESTION: "агент нечітко пояснює різницю тарифів або наслідки зміни плану",
            IssueType.REFUND_REQUEST: "агент нечітко або помилково пояснює політику повернення коштів",
        }
        error_templates = {
            CaseClass.PROBLEMATIC: [
                AgentError(
                    type=AgentErrorType.PROCESS,
                    description=issue_specific[issue_type],
                    severity=3,
                ),
            ],
            CaseClass.CONFLICT: [
                AgentError(
                    type=AgentErrorType.TONE,
                    description="агент відповідає занадто різко або захисно замість зниження напруги",
                    severity=4,
                ),
                AgentError(
                    type=AgentErrorType.EMPATHY,
                    description="агент недостатньо визнає реальний дискомфорт клієнта",
                    severity=4,
                ),
            ],
            CaseClass.AGENT_MISTAKE: [
                AgentError(
                    type=AgentErrorType.LOGIC,
                    description=issue_specific[issue_type],
                    severity=4,
                ),
                AgentError(
                    type=AgentErrorType.TONE if rng.random() < 0.5 else AgentErrorType.ACCURACY,
                    description="агент формулює відповідь так, що вводить клієнта в оману або дратує його",
                    severity=3,
                ),
            ],
        }
        return error_templates.get(case_class, [])

