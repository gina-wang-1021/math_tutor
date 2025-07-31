import os
import unittest
import sys

# NOT UPDATED

# Add project root and scripts directory to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))

from utilities.qa_utils import rephrase_question, fetch_historic_data, store_new_data

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

    def test_fetch_historic_data(self):
        """Test the historic data fetching function."""
        question_one = "What is 2+2?"
        question_two = "What does triangle have?"
        question_three = "What is 3+3?"
        question_four = "how do I code python"

        fetch_result_one, id_one = fetch_historic_data(question_one, True)
        fetch_result_two, id_two = fetch_historic_data(question_two, False)
        fetch_result_three, id_three = fetch_historic_data(question_three, True)
        fetch_result_four, id_four = fetch_historic_data(question_four, True)
        
        self.assertEqual(fetch_result_one, "2+2 is 4")
        self.assertIsInstance(id_one, int)
        self.assertEqual(fetch_result_two, "Triangle has 3 sides")
        self.assertIsInstance(id_two, int)
        self.assertEqual(fetch_result_three, "3+3 is 6")
        self.assertIsInstance(id_three, int)
        self.assertEqual(fetch_result_four, None)
        self.assertEqual(id_four, None)

        fetch_result_one_false, id_one_false = fetch_historic_data(question_one, False)
        self.assertEqual(fetch_result_one_false, None)
        self.assertEqual(id_one_false, 13)
        
    
    def test_store_new_data(self):
        """Test the new data storage function."""
        question_one = "What is 2+2?"
        question_two = "What does triangle have?"
        question_three = "What is 3+3?"

        # storage_result_one = store_new_data(question_one, "2+2 is 4", True)
        # storage_result_two = store_new_data(question_two, "Triangle has 3 sides", False)
        # storage_result_three = store_new_data(question_three, "3+3 is 6", True)
        storage_result_four = store_new_data(question_one, "2+2 is 4 yup yup", False, 13)

        # self.assertTrue(storage_result_one)
        # self.assertTrue(storage_result_two)
        # self.assertTrue(storage_result_three)
        self.assertTrue(storage_result_four)

if __name__ == '__main__':
    unittest.main()
