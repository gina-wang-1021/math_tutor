import os
import unittest
import sys

# Add project root and scripts directory to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))

from engine import pipeline

class TestPipeline(unittest.TestCase):
    
    def test_calculation_question(self):
        """Test the pipeline with a calculation question."""
        student_id = "1"  # Using an existing student ID
        question = "What is 25 divided by 5?"
        history = ""  # No previous conversation
        
        response = pipeline(student_id, question, history)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # The answer should contain the number 5 somewhere
        self.assertIn("5", response)
    
    def test_overview_question(self):
        """Test the pipeline with an overview question."""
        student_id = "1"
        question = "What topics can you help me with?"
        history = ""
        
        response = pipeline(student_id, question, history)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_topic_based_question(self):
        """Test the pipeline with a topic-based question."""
        student_id = "1"
        question = "What is a quadratic equation?"
        history = ""
        
        print("generating response")
        response = pipeline(student_id, question, history)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_follow_up_question(self):
        """Test the pipeline with a follow-up question that requires context."""
        student_id = "1"
        question = "How do I solve them?"
        history = "Human: What are quadratic equations?\nAI: Quadratic equations are equations of the form ax² + bx + c = 0 where a ≠ 0."
        
        response = pipeline(student_id, question, history)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
    
    def test_nonexistent_student(self):
        """Test the pipeline with a nonexistent student ID."""
        student_id = "nonexistent_student"
        question = "What is a triangle?"
        history = ""
        
        response = pipeline(student_id, question, history)
        
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()
