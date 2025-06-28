import os
import unittest
import sys

# Add project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utilities.qa_utils import get_historic_answer, rephrase_question, MAX_L2_DISTANCE_THRESHOLD

class TestQaUtils(unittest.TestCase):

    def test_rephrase_question(self):
        """Test the rephrasing function with a real LLM call."""
        history = "Human: What are polynomials?\nAI: They are expressions with variables and coefficients."
        user_question = "what about their degrees?"
        standalone_question = rephrase_question(user_question, history)
        self.assertIsInstance(standalone_question, str)
        self.assertNotIn("their", standalone_question.lower())
        self.assertIn("degree", standalone_question.lower())
        self.assertIn("polynomial", standalone_question.lower())

    def test_get_historic_answer_found(self):
        """Test finding a historic answer from FAISS/SQLite."""
        # This test assumes a similar question about the quadratic formula exists for grade 11.
        question = "How do you use the quadratic formula?"
        grade = 11
        # Use a slightly higher threshold to ensure a match for this integration test
        historic_answer = get_historic_answer(grade, question, max_distance_threshold=0.5)
        self.assertIsNotNone(historic_answer, "Should find a historic answer for a known topic.")
        self.assertIsInstance(historic_answer, str)

    def test_get_historic_answer_not_found(self):
        """Test that no historic answer is found for a novel question."""
        question = "What is the philosophical implication of the Riemann hypothesis?"
        grade = 12
        historic_answer = get_historic_answer(grade, question, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD)
        self.assertIsNone(historic_answer, "Should not find a historic answer for a novel question.")

    def test_get_historic_answer_wrong_grade(self):
        """Test that no historic answer is found for a grade without historic data."""
        question = "What is 2+2?"
        grade = 10  # No historic data for grade 10
        historic_answer = get_historic_answer(grade, question, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD)
        self.assertIsNone(historic_answer, "Should not attempt to find answers for grades without data.")

if __name__ == '__main__':
    unittest.main()
