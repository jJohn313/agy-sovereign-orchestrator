import unittest
import json
from orchestrator import AgyOrchestrator, ToolCall, AgentResponse

class TestHistoricalCompaction(unittest.TestCase):
    def test_compaction_in_agent_loop(self):
        orchestrator = AgyOrchestrator()

        # We will mock `call_frontier_model` to simulate a 4-step loop
        # Turn 1: tool call A -> success
        # Turn 2: tool call B -> error
        # Turn 3: tool call C -> success (T-1)
        # Turn 4: Final response

        call_counter = 0
        frontier_payloads = []

        def mock_call_frontier_model(messages, tools, context):
            nonlocal call_counter

            # Save the exact payload received by the frontier model
            import copy
            frontier_payloads.append(copy.deepcopy(messages))

            call_counter += 1
            if call_counter == 1:
                # Turn 1 tool call
                return AgentResponse(
                    content="",
                    tool_calls=[ToolCall(name="ToolA", arguments={"arg": 1}, call_id="call_A")]
                )
            elif call_counter == 2:
                # Turn 2 tool call
                return AgentResponse(
                    content="",
                    tool_calls=[ToolCall(name="ToolB", arguments={"arg": 2}, call_id="call_B")]
                )
            elif call_counter == 3:
                # Turn 3 tool call
                return AgentResponse(
                    content="",
                    tool_calls=[ToolCall(name="ToolC", arguments={"arg": 3}, call_id="call_C")]
                )
            else:
                # Final response
                return AgentResponse(content="Final result achieved.")

        def mock_execute_tool(call: ToolCall) -> str:
            if call.name == "ToolA":
                # Success, long output
                return "Line 1\nLine 2\nLine 3\n" * 20 + "OK done."
            elif call.name == "ToolB":
                # Error anchor output
                return "Starting...\nTraceback (most recent call last):\n  File 'test.py', line 1\nValueError: Test error\nEnd"
            elif call.name == "ToolC":
                # Success, short output
                return "Tool C completed successfully."
            return ""

        result = orchestrator.run_agent_loop(
            prompt="Test prompt",
            active_tools=[],
            context={},
            call_frontier_model=mock_call_frontier_model,
            execute_tool=mock_execute_tool
        )

        internal_messages = result["messages"]

        # Verify internal history remains full
        tool_a_raw_found = False
        tool_b_raw_found = False
        tool_c_raw_found = False
        for msg in internal_messages:
            if msg.get("role") == "tool":
                if msg.get("name") == "ToolA":
                    self.assertGreater(len(msg["content"]), 300)
                    tool_a_raw_found = True
                elif msg.get("name") == "ToolB":
                    self.assertIn("Traceback", msg["content"])
                    self.assertGreater(len(msg["content"]), 80)
                    tool_b_raw_found = True
                elif msg.get("name") == "ToolC":
                    self.assertIn("Tool C completed successfully", msg["content"])
                    tool_c_raw_found = True

        self.assertTrue(tool_a_raw_found)
        self.assertTrue(tool_b_raw_found)
        self.assertTrue(tool_c_raw_found)

        # Let's examine the payload received by the frontier model on the *final* call (Turn 4)
        # It should have ToolA and ToolB compressed, but ToolC uncompressed.
        final_payload = frontier_payloads[-1]

        tool_a_compressed_found = False
        tool_b_compressed_found = False
        tool_c_compressed_found = False

        for msg in final_payload:
            if msg.get("role") == "tool":
                if msg.get("name") == "ToolA":
                    self.assertLess(len(msg["content"]), 250)
                    data = json.loads(msg["content"])
                    self.assertEqual(data["tool"], "ToolA")
                    self.assertEqual(data["status"], "ok")
                    tool_a_compressed_found = True
                elif msg.get("name") == "ToolB":
                    self.assertLess(len(msg["content"]), 250)
                    data = json.loads(msg["content"])
                    self.assertEqual(data["tool"], "ToolB")
                    self.assertEqual(data["status"], "err")
                    self.assertIn("Traceback", data["summary"])
                    tool_b_compressed_found = True
                elif msg.get("name") == "ToolC":
                    # T-1 tool call, should NOT be compressed
                    # Our execution string is "Tool C completed successfully." (but fold_log_output prepends a checkmark if exit code 0)
                    # We just assert it is NOT a JSON stub
                    try:
                        json.loads(msg["content"])
                        # If it parses as JSON, it might have been incorrectly compressed, unless fold_log_output outputs json (it doesn't by default here)
                        # Let's ensure it doesn't match our stub format
                        if "status" in json.loads(msg["content"]):
                            self.fail("ToolC was incorrectly compressed into a stub!")
                    except json.JSONDecodeError:
                        pass # expected
                    tool_c_compressed_found = True

        self.assertTrue(tool_a_compressed_found)
        self.assertTrue(tool_b_compressed_found)
        self.assertTrue(tool_c_compressed_found)

if __name__ == "__main__":
    unittest.main()
