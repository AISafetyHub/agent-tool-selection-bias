"""smolagents-based evaluation adapter.

Wraps scenario tools into smolagents Tool subclasses for comparison.
Produces the same output format as simulator.py.
"""


def simulate_smolagent(
    scenario: dict,
    model: str,
    system_prompt: str,
    max_turns: int = 5,
    error_strategy: str = "first_attempt_fail",
) -> dict:
    """Same interface as simulator.simulate() but using smolagents.

    TODO: Implement after custom simulator is validated.
    """
    raise NotImplementedError("smolagents adapter not yet implemented")
