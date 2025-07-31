import os
import unittest
import sys

# Add project root and scripts directory to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'scripts'))

from utilities.student_utils import get_student_level, get_confidence_level_and_score, check_confidence_and_score

class TestStudentUtils(unittest.TestCase):

    def test_get_student_level_existing(self):
        """Test retrieving the level for an existing student."""
        # Sample student data dictionary
        student_data = {
            'Student ID': '1',
            'First name': 'Test',
            'Last name': 'Student',
            'Current Class': '12',
            'Stream': 'Science with Mathematics',
            'Score': 'A (64-71)'
        }
        vectorstore, tablespace, grade = get_student_level(student_data)
        self.assertEqual(vectorstore, 'grade-twelve-math')
        self.assertEqual(tablespace, 'gradeTwelveMath')
        self.assertEqual(grade, 12)

    def test_get_student_level_missing_data(self):
        """Test retrieving the level with missing data."""
        # Test with missing Current Class key
        student_data = {
            'Student ID': '1',
            'First name': 'Test',
            'Last name': 'Student'
        }
        vectorstore, tablespace, grade = get_student_level(student_data)
        # Should return default grade 11 values
        self.assertEqual(vectorstore, 'grade-eleven-math')
        self.assertEqual(tablespace, 'gradeElevenMath')
        self.assertEqual(grade, 11)

    def test_get_confidence_level_and_score(self):
        """Test retrieving confidence level and score for a student and topic."""
        # Sample student data dictionary with algebra confidence level
        student_data = {
            'Student ID': '1',
            'First name': 'Test',
            'Last name': 'Student',
            'Current Class': '12',
            'Stream': 'Science with Mathematics',
            'Score': 'A (64-71)',
            'Algebra': '4 = Confident'
        }
        level, score = get_confidence_level_and_score(student_data, 'algebra')
        self.assertIn(level, [1, 2, 3, 4, 5])
        self.assertIsInstance(score, str)
        self.assertEqual(level, 4)  # Should be 4 based on '4 = Confident'
        self.assertEqual(score, 'A')  # Should be 'A' based on 'A (64-71)'

    def test_check_confidence_and_score(self):
        """Test the confidence and score checking logic."""
        self.assertTrue(check_confidence_and_score(4, "A"))
        self.assertFalse(check_confidence_and_score(1, "B"))
        self.assertFalse(check_confidence_and_score(5, "C"))

if __name__ == '__main__':
    unittest.main()
