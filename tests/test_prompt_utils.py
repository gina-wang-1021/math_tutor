import os
import unittest
import sys

# Add project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utilities.prompt_utils import load_prompt

class TestPromptUtils(unittest.TestCase):

    def test_load_existing_prompt(self):
        """Test loading an existing prompt file."""
        prompt_content = load_prompt("compare_prompt.txt")
        self.assertIsInstance(prompt_content, str)
        self.assertIn("Compare the two answers", prompt_content)

    def test_load_nonexistent_prompt(self):
        """Test loading a nonexistent prompt file raises an error."""
        with self.assertRaises(FileNotFoundError):
            load_prompt("nonexistent_prompt.txt")

if __name__ == '__main__':
    unittest.main()
