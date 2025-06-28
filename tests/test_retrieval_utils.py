import os
import unittest
import sys

# Add project root to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Set OpenAI API key from .env file
env_path = os.path.join(PROJECT_ROOT, '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.strip().split('=', 1)[1].strip('"\'')
                break

from utilities.retrieval_utils import get_topic, get_chunks_for_current_year, get_chunks_from_prior_years

class TestRetrievalUtils(unittest.TestCase):

    def test_get_topic(self):
        """Test the topic detection functionality with a real LLM call."""
        question = "What is a quadratic equation?"
        topic = get_topic(question)
        self.assertIsInstance(topic, str)
        self.assertIn(topic.lower(), ["algebra", "geometry", "basics", "modelling"])

    def test_get_chunks_for_current_year(self):
        """Test retrieving chunks for the current year from Chroma DB."""
        # This test assumes the 'algebra' index exists for the 'beginner' level
        topic = "geometry"
        level = "beginner"
        question = "What is a triangle?"
        chunks = get_chunks_for_current_year(topic, level, question)
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0, "Should retrieve at least one chunk for a valid topic")
        if chunks:
            self.assertIsInstance(chunks[0], str)

    def test_get_chunks_from_prior_years(self):
        """Test retrieving chunks from prior years from Chroma DB."""
        # This test assumes the 'algebra' index exists for prior levels
        topic = "algebra"
        current_level = "advanced"
        chunks = get_chunks_from_prior_years(topic, current_level, "How do I solve a quadratic equation?")
        self.assertIsInstance(chunks, list)
        self.assertTrue(len(chunks) > 0, "Should retrieve chunks from prior years for an advanced student")
        if chunks:
            self.assertIsInstance(chunks[0], str)

    def test_get_chunks_for_beginner_no_prior(self):
        """Test that no prior year chunks are retrieved for a beginner."""
        topic = "geometry"
        current_level = "beginner"
        chunks = get_chunks_from_prior_years(topic, current_level, "What is a triangle?")
        self.assertIsInstance(chunks, list)
        self.assertEqual(len(chunks), 0, "Beginners should not have prior year chunks")

if __name__ == '__main__':
    unittest.main()
