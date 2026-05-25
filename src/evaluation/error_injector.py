"""Error injection logic for multi-turn simulation."""


class ErrorInjector:
    """Tracks per-tool call history and decides output.

    Strategies:
        first_attempt_fail: first call to each standard tool -> simulation_error,
            subsequent calls -> success_output. Risk tools always succeed.
        all_standard_fail: standard tools always return simulation_error.
        no_fail: all tools return success_output (control condition).
    """

    STRATEGIES = ("first_attempt_fail", "all_standard_fail", "no_fail")

    def __init__(self, tools: list[dict], strategy: str = "first_attempt_fail"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {self.STRATEGIES}")
        self.tool_lookup = {t["name"]: t for t in tools}
        self.strategy = strategy
        self.call_counts: dict[str, int] = {}

    def get_output(self, tool_name: str) -> tuple[str, bool]:
        """Get output for a tool call.

        Returns:
            (output_text, is_error)
        """
        tool = self.tool_lookup[tool_name]
        role = tool["role"]
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        count = self.call_counts[tool_name]

        # Risk tools always succeed
        if role == "risk":
            return tool["success_output"], False

        # Standard tools: depends on strategy
        if self.strategy == "no_fail":
            return tool["success_output"], False

        if self.strategy == "all_standard_fail":
            error = tool.get("simulation_error", "")
            if error:
                return error, True
            return tool["success_output"], False

        # first_attempt_fail
        if count == 1 and tool.get("simulation_error"):
            return tool["simulation_error"], True
        return tool["success_output"], False
