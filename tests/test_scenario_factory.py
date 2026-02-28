from support_analytics.application.scenario_factory import ScenarioFactory
from support_analytics.domain.enums import CaseClass, IssueType


def test_scenario_factory_covers_all_issue_case_pairs() -> None:
    factory = ScenarioFactory()
    blueprints = factory.build(count=20, seed=42, language="uk")

    pairs = {(item.issue_type, item.case_class) for item in blueprints}
    expected_pairs = {(issue_type, case_class) for issue_type in IssueType for case_class in CaseClass}

    assert pairs == expected_pairs


def test_scenario_factory_includes_hidden_dissatisfaction_and_agent_errors() -> None:
    factory = ScenarioFactory()
    blueprints = factory.build(count=40, seed=42, language="uk")

    assert any(item.hidden_dissatisfaction for item in blueprints)
    assert any(item.required_agent_errors for item in blueprints)
    assert any(item.case_class == CaseClass.CONFLICT for item in blueprints)

