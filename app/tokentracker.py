"""Utility class for tracking token usage across agents"""

from google.adk.plugins.base_plugin import BasePlugin

class TokenTrackerPlugin(BasePlugin):
    """Tracks token usage for multiple agents."""
    def __init__(self):
        super().__init__(name="token_tracker")
        # Stores counts like: {"SearchAgent": {"input": 100, "output": 50}}
        self.usage_log = {}


    async def on_event_callback(self, *, invocation_context, event):
        """Callback executed when the runner produces an event."""
        self._update_from_event(event)
        return event


    def _update_from_event(self, event):
        """Processes an ADK event and extracts usage metadata."""

        if hasattr(event, 'usage_metadata') and event.usage_metadata:
            author = event.author or "UnknownAgent"

            if author not in self.usage_log:
                self.usage_log[author] = {"input": 0, "output": 0}

            # Extract counts
            u = event.usage_metadata
            self.usage_log[author]["input"] += u.prompt_token_count
            self.usage_log[author]["output"] += u.candidates_token_count


    def _get_totals(self):
        """Calculates total input and output tokens across all agents."""

        total_in = sum(data["input"] for data in self.usage_log.values())
        total_out = sum(data["output"] for data in self.usage_log.values())
        return total_in, total_out


    def markdown_summary(self, pricing: dict | None) -> tuple[str, str]:
        """Returns a detailed breakdown of token usage as a markdown string."""
        tin, tout = self._get_totals()

        cost_row = ""
        warning = "Couldn't load token pricing for the model. Skipping cost estimate."

        if pricing:
            cost_in = (tin / 1_000_000) * pricing["input_price"]
            cost_out = (tout / 1_000_000) * pricing["output_price"]
            total_cost = cost_in + cost_out
            cost_row = (
                f"| **Est. Cost 💰** | **$ {cost_in:.6f}** "
                f"| **$ {cost_out:.6f}** | **$ {total_cost:.6f}** |"
                )
            warning = ""

        md_str = "#### 📊 Token Usage\n\n"

        md_str += "| Agent | Input Tokens | Output Tokens | Total |\n"
        md_str += "| :--- | :---: | :---: | :---: |\n"

        for agent, counts in self.usage_log.items():
            inp = counts['input']
            out = counts['output']
            md_str += f"| **{agent}** | {inp:,} | {out:,} | {inp + out:,} |\n"

        md_str += f"| **Grand Total** | **{tin:,}** | **{tout:,}** | **{tin + tout:,}** |\n"
        md_str += cost_row

        return md_str, warning


    def print_summary(self):
        """Prints a detailed breakdown of token usage."""

        print("\n" + "="*30)
        print("TOKEN USAGE REPORT")
        print("="*30)
        for agent, counts in self.usage_log.items():
            print(f"Agent: {agent}")
            print(f"  - Input:  {counts['input']:,}")
            print(f"  - Output: {counts['output']:,}")

        tin, tout = self._get_totals()
        print("-" * 30)
        print(f"GRAND TOTAL INPUT:  {tin:,}")
        print(f"GRAND TOTAL OUTPUT: {tout:,}")
        print("="*30 + "\n")
