import os
import unittest
import sys

# Add project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from utilities.student_utils import get_student_level, get_confidence_level_and_score, check_confidence_and_score

class TestStudentUtils(unittest.TestCase):

    def test_get_student_level_existing(self):
        """Test retrieving the level for an existing student."""
        # This test assumes a student with ID '1' exists in the data
        level, grade = get_student_level('1')
        self.assertEqual(level, 'advanced')
        self.assertEqual(grade, 12)

    def test_get_student_level_nonexistent(self):
        """Test retrieving the level for a nonexistent student."""
        level, grade = get_student_level('nonexistent_student')
        self.assertIsNone(level, None)
        self.assertIsNone(grade, None)

    def test_get_confidence_level_and_score(self):
        """Test retrieving confidence level and score for a student and topic."""
        # This test assumes student '1' has data for the 'algebra' topic
        level, score = get_confidence_level_and_score('1', 'algebra')
        self.assertIn(level, [1, 2, 3, 4, 5])
        self.assertIsInstance(score, str)

    def test_check_confidence_and_score(self):
        """Test the confidence and score checking logic."""
        self.assertTrue(check_confidence_and_score(4, "A"))
        self.assertFalse(check_confidence_and_score(1, "B"))
        self.assertFalse(check_confidence_and_score(5, "C"))

if __name__ == '__main__':
    unittest.main()
