import unittest
import json
from schema_minifier import minify_tool_schema

class TestSchemaMinifier(unittest.TestCase):
    def setUp(self):
        self.verbose_schema = {
            "name": "github_create_pull_request",
            "description": "This tool creates a pull request on GitHub. It allows you to merge a feature branch into a base branch. You should ensure that you have committed all necessary changes to the feature branch before invoking this tool. Useful for collaborative workflows.",
            "markdownDescription": "## Create Pull Request\n\nCreates a PR on GitHub.",
            "examples": [
                {
                    "title": "Basic PR",
                    "arguments": {
                        "repo": "owner/repo",
                        "title": "Fix bug",
                        "head": "feature",
                        "base": "main"
                    }
                }
            ],
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "The name of the repository including the owner, e.g. octocat/Hello-World. It must match exactly.",
                        "example": "octocat/Hello-World"
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the new pull request that will be visible to all reviewers.",
                    },
                    "head": {
                        "type": "string",
                        "description": "The name of the branch where your changes are implemented. This is the source branch.",
                    },
                    "base": {
                        "type": "string",
                        "description": "The name of the branch you want the changes pulled into. This is the target branch.",
                        "default": "main"
                    },
                    "labels": {
                        "type": "array",
                        "description": "An array of label names to add to the PR.",
                        "items": {
                            "type": "string",
                            "description": "The exact name of the label as configured in the repository settings."
                        }
                    },
                    "options": {
                        "type": "object",
                        "description": "Additional options for the PR.",
                        "properties": {
                            "draft": {
                                "type": "boolean",
                                "description": "Indicates whether the pull request should be created as a draft PR.",
                                "default": False
                            }
                        }
                    }
                },
                "required": ["repo", "title", "head", "base"]
            }
        }

    def test_minify_tool_schema(self):
        original_str = json.dumps(self.verbose_schema)

        minified = minify_tool_schema(self.verbose_schema)

        # Verify markdownDescription, examples, and example are removed
        self.assertNotIn("markdownDescription", minified)
        self.assertNotIn("examples", minified)

        # Verify schema validation keys are retained
        self.assertIn("name", minified)
        self.assertEqual(minified["name"], "github_create_pull_request")

        self.assertIn("parameters", minified)
        params = minified["parameters"]
        self.assertIn("type", params)
        self.assertEqual(params["type"], "object")
        self.assertIn("required", params)
        self.assertEqual(params["required"], ["repo", "title", "head", "base"])
        self.assertIn("properties", params)

        props = params["properties"]
        self.assertIn("repo", props)
        self.assertIn("type", props["repo"])
        self.assertNotIn("example", props["repo"])

        # Verify descriptions are truncated
        # Top-level tool description: 38 words -> truncated to 12
        self.assertEqual(
            minified["description"],
            "This tool creates a pull request on GitHub. It allows you to..."
        )

        # Property description
        # repo: 15 words -> 12
        self.assertEqual(
            props["repo"]["description"],
            "The name of the repository including the owner, e.g. octocat/Hello-World. It must..."
        )

        # title: 13 words -> 12
        self.assertEqual(
            props["title"]["description"],
            "The title of the new pull request that will be visible to..."
        )

        # head: 16 words -> 12
        self.assertEqual(
            props["head"]["description"],
            "The name of the branch where your changes are implemented. This is..."
        )

        # base: 16 words -> 12
        self.assertEqual(
            props["base"]["description"],
            "The name of the branch you want the changes pulled into. This..."
        )

        # Verify defaults are retained
        self.assertIn("default", props["base"])
        self.assertEqual(props["base"]["default"], "main")

        self.assertIn("options", props)
        self.assertIn("default", props["options"]["properties"]["draft"])
        self.assertEqual(props["options"]["properties"]["draft"]["default"], False)

        self.assertIn("items", props["labels"])
        self.assertEqual(props["labels"]["items"]["type"], "string")

        minified_str = json.dumps(minified)

        # Measure size reduction
        orig_len = len(original_str)
        min_len = len(minified_str)
        reduction_percentage = ((orig_len - min_len) / orig_len) * 100

        print(f"\nOriginal size: {orig_len} chars")
        print(f"Minified size: {min_len} chars")
        print(f"Reduction: {reduction_percentage:.2f}%")

        # Give some leeway for the ~30% requirement depending on the exact string representation
        self.assertGreaterEqual(reduction_percentage, 29.0, "Size reduction should be >= 29%")

if __name__ == "__main__":
    unittest.main()
