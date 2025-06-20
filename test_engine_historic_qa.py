import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np
import os

# Assuming engine.py is in the same directory or accessible via PYTHONPATH
from engine import pipeline, MAX_L2_DISTANCE_THRESHOLD, HISTORIC_QA_K_NEIGHBORS

# Mock faiss and sqlite3 if they are not available in the test environment or to control their behavior
# For simplicity, we'll often mock the functions that use them directly from engine.py

class TestHistoricQARetrieval(unittest.TestCase):

    def setUp(self):
        """Set up common mock objects and configurations for tests."""
        self.student_id_grade11 = "student_g11"
        self.student_id_grade12 = "student_g12"
        self.student_id_other_grade = "student_g10"
        self.user_question = "What is calculus?"
        self.history = ""
        self.mock_rephrased_question = "Explain calculus in detail."
        self.mock_historic_answer = "Calculus is a branch of mathematics."
        self.mock_normal_pipeline_answer = "This is a new answer about calculus."

        # Mock the embedding for the rephrased question
        self.mock_question_embedding = np.array([[0.1, 0.2, 0.3]], dtype='float32')

        # Default L2 distance for a 'good' match (below threshold)
        self.good_match_distance = MAX_L2_DISTANCE_THRESHOLD / 2
        # Default L2 distance for a 'bad' match (above threshold)
        self.bad_match_distance = MAX_L2_DISTANCE_THRESHOLD * 2
        self.match_id = np.array([[123]], dtype='int64')

    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Explain calculus in detail.')
    @patch('engine.OpenAIEmbeddings') # Mocks the class
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa')
    @patch('engine.get_topic') # Mock to prevent further pipeline execution if historic fails
    def test_historic_answer_found_grade11(
        self, mock_get_topic, mock_search_historic_qa, mock_load_historic_qa_resources, 
        mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test successful retrieval of a historic answer for a Grade 11 student."""
        # Setup mocks
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()
        
        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        mock_search_historic_qa.return_value = self.mock_historic_answer

        # Call pipeline
        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        # Assertions
        mock_get_student_level.assert_called_once_with(self.student_id_grade11)
        mock_rephrase_question.assert_called_once_with(self.user_question, self.history)
        mock_openai_embeddings_instance.embed_query.assert_called_once_with(self.mock_rephrased_question)
        mock_load_historic_qa_resources.assert_called_once_with(11)
        mock_search_historic_qa.assert_called_once_with(
            unittest.mock.ANY, # np.array comparison can be tricky, check shape/dtype if needed
            mock_faiss_index, 
            mock_db_conn, 
            k=HISTORIC_QA_K_NEIGHBORS, 
            max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )
        self.assertEqual(result, f"This question is similar to one asked before. Here's the previous answer: {self.mock_historic_answer}")
        mock_get_topic.assert_not_called() # Pipeline should short-circuit

    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Explain calculus for grade 12.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa')
    @patch('engine.get_topic') 
    def test_historic_answer_found_grade12(
        self, mock_get_topic, mock_search_historic_qa, mock_load_historic_qa_resources, 
        mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test successful retrieval of a historic answer for a Grade 12 student."""
        mock_get_student_level.return_value = ("advanced", 12)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()
        
        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        mock_search_historic_qa.return_value = self.mock_historic_answer

        result = pipeline(self.student_id_grade12, self.user_question, self.history)

        mock_load_historic_qa_resources.assert_called_once_with(12)
        mock_search_historic_qa.assert_called_once_with(
            unittest.mock.ANY, mock_faiss_index, mock_db_conn, 
            k=HISTORIC_QA_K_NEIGHBORS, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )
        self.assertEqual(result, f"This question is similar to one asked before. Here's the previous answer: {self.mock_historic_answer}")
        mock_get_topic.assert_not_called()

    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='A slightly different calculus question.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa', return_value=None) # search_historic_qa returns None if threshold not met
    @patch('engine.get_topic', return_value="calculus_topic") # Assume normal pipeline continues
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk1", metadata={'level':'advanced'})])
    @patch('engine.LLMChain') # Mock the final answer generation chain
    def test_historic_answer_not_found_due_to_threshold(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa, 
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test pipeline proceeds normally when historic answer similarity is below threshold."""
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()

        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        
        # Simulate the LLMChain for normal answer generation
        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        mock_search_historic_qa.assert_called_once() # search_historic_qa was called
        mock_get_topic.assert_called_once_with('A slightly different calculus question.') # Normal pipeline continues
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='A very unique calculus question.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa', return_value=None) # search_historic_qa returns None if FAISS finds no good match
    @patch('engine.get_topic', return_value="calculus_unique") 
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_unique", metadata={'level':'advanced'})])
    @patch('engine.LLMChain')
    def test_historic_answer_not_found_no_faiss_match(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa, 
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test pipeline proceeds normally when FAISS finds no similar historic question."""
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()

        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        
        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        mock_search_historic_qa.assert_called_once() # search_historic_qa was called
        mock_get_topic.assert_called_once_with('A very unique calculus question.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='A basic math question.')
    @patch('engine.load_historic_qa_resources') # Should not be called
    @patch('engine.OpenAIEmbeddings') # Should not be called for historic part
    @patch('engine.get_topic', return_value="basics")
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_basics", metadata={'level':'beginner'})])
    @patch('engine.LLMChain')
    def test_historic_qa_skipped_for_other_grades(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_openai_embeddings_class, 
        mock_load_historic_qa_resources, mock_rephrase_question, mock_get_student_level):
        """Test historic Q&A is skipped for students not in Grade 11 or 12."""
        mock_get_student_level.return_value = ("beginner", 10) # e.g., Grade 10
        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value # To check calls

        result = pipeline(self.student_id_other_grade, self.user_question, self.history)

        mock_load_historic_qa_resources.assert_not_called()
        mock_openai_embeddings_instance.embed_query.assert_not_called() # embed_query for historic Q&A part
        mock_get_topic.assert_called_once_with('A basic math question.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Another calculus question.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources', return_value=(None, MagicMock())) # FAISS index load fails
    @patch('engine.search_historic_qa') # Should not be called if load fails
    @patch('engine.get_topic', return_value="calculus_topic_2")
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_calc2", metadata={'level':'advanced'})])
    @patch('engine.LLMChain')
    def test_historic_qa_resource_load_faiss_fails(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa, 
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test pipeline proceeds normally if loading FAISS index fails."""
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()

        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        mock_load_historic_qa_resources.assert_called_once_with(11)
        mock_search_historic_qa.assert_not_called() # Search should not happen if resources don't load
        mock_get_topic.assert_called_once_with('Another calculus question.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)

    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Question needing DB.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources', return_value=(MagicMock(), None)) # DB connection fails
    @patch('engine.search_historic_qa') 
    @patch('engine.get_topic', return_value="db_fail_topic")
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_db_fail", metadata={'level':'advanced'})])
    @patch('engine.LLMChain')
    def test_historic_qa_resource_load_db_fails(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa, 
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test pipeline proceeds normally if loading SQLite DB fails."""
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()

        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        mock_load_historic_qa_resources.assert_called_once_with(11)
        mock_search_historic_qa.assert_not_called() # Search should not happen if DB conn fails (part of resource load)
        mock_get_topic.assert_called_once_with('Question needing DB.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Calculus question for custom threshold.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa') # We will assert its call with the custom threshold
    @patch('engine.get_topic')
    def test_historic_qa_configurable_threshold_allows_retrieval(
        self, mock_get_topic, mock_search_historic_qa, mock_load_historic_qa_resources,
        mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test a looser custom threshold allows historic answer retrieval."""
        custom_loose_threshold = MAX_L2_DISTANCE_THRESHOLD * 1.5 # Looser than default
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()
        
        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        # search_historic_qa will be called with custom_loose_threshold, and we make it return an answer
        mock_search_historic_qa.return_value = self.mock_historic_answer 

        result = pipeline(self.student_id_grade11, self.user_question, self.history, historic_qa_l2_threshold=custom_loose_threshold)

        mock_search_historic_qa.assert_called_once_with(
            unittest.mock.ANY, mock_faiss_index, mock_db_conn, 
            k=HISTORIC_QA_K_NEIGHBORS, max_distance_threshold=custom_loose_threshold
        )
        self.assertEqual(result, f"This question is similar to one asked before. Here's the previous answer: {self.mock_historic_answer}")
        mock_get_topic.assert_not_called()


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Calculus question for stricter threshold.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa') # We will assert its call with the custom threshold
    @patch('engine.get_topic', return_value="calculus_strict")
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_strict", metadata={'level':'advanced'})])
    @patch('engine.LLMChain')
    def test_historic_qa_configurable_threshold_rejects_retrieval(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa,
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test a stricter custom threshold rejects historic answer retrieval."""
        custom_strict_threshold = MAX_L2_DISTANCE_THRESHOLD / 2 # Stricter than default
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()
        
        mock_faiss_index = MagicMock()
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index, mock_db_conn)
        # search_historic_qa will be called with custom_strict_threshold, and we make it return None (as if match failed threshold)
        mock_search_historic_qa.return_value = None

        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history, historic_qa_l2_threshold=custom_strict_threshold)

        mock_search_historic_qa.assert_called_once_with(
            unittest.mock.ANY, mock_faiss_index, mock_db_conn, 
            k=HISTORIC_QA_K_NEIGHBORS, max_distance_threshold=custom_strict_threshold
        )
        mock_get_topic.assert_called_once_with('Calculus question for stricter threshold.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.get_student_level')
    @patch('engine.rephrase_question', return_value='Question for empty index.')
    @patch('engine.OpenAIEmbeddings')
    @patch('engine.load_historic_qa_resources')
    @patch('engine.search_historic_qa') # We will effectively test its internal logic via load_historic_qa_resources and FAISS mock
    @patch('engine.get_topic', return_value="empty_index_topic")
    @patch('engine.get_relevant_chunks', return_value=[MagicMock(page_content="chunk_empty", metadata={'level':'advanced'})])
    @patch('engine.LLMChain')
    def test_pipeline_with_empty_faiss_index_handling(
        self, mock_llm_chain, mock_get_relevant_chunks, mock_get_topic, mock_search_historic_qa,
        mock_load_historic_qa_resources, mock_openai_embeddings_class, mock_rephrase_question, mock_get_student_level):
        """Test pipeline proceeds normally if FAISS index is empty (ntotal=0)."""
        mock_get_student_level.return_value = ("intermediate", 11)
        mock_openai_embeddings_instance = mock_openai_embeddings_class.return_value
        mock_openai_embeddings_instance.embed_query.return_value = self.mock_question_embedding.flatten().tolist()
        
        # Simulate search_historic_qa returning None because the index was empty
        # This happens if faiss_index.ntotal == 0 inside search_historic_qa
        mock_faiss_index_empty = MagicMock()
        mock_faiss_index_empty.ntotal = 0
        mock_db_conn = MagicMock()
        mock_load_historic_qa_resources.return_value = (mock_faiss_index_empty, mock_db_conn)
        
        # search_historic_qa should return None if index is empty before attempting search
        mock_search_historic_qa.return_value = None 

        mock_answer_chain_instance = mock_llm_chain.return_value
        mock_answer_chain_instance.run.return_value = self.mock_normal_pipeline_answer

        result = pipeline(self.student_id_grade11, self.user_question, self.history)

        mock_load_historic_qa_resources.assert_called_once_with(11)
        # We expect search_historic_qa to be called, and it should internally handle ntotal=0 returning None.
        mock_search_historic_qa.assert_called_once_with(
            unittest.mock.ANY, mock_faiss_index_empty, mock_db_conn, 
            k=HISTORIC_QA_K_NEIGHBORS, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )
        mock_get_topic.assert_called_once_with('Question for empty index.')
        self.assertEqual(result, self.mock_normal_pipeline_answer)


    @patch('engine.sqlite3.connect') # Mock the actual sqlite3.connect call
    def test_search_historic_qa_helper_finds_match(self, mock_sqlite_connect):
        """Test search_historic_qa helper directly: finds a match."""
        from engine import search_historic_qa # Import locally for clarity

        mock_faiss_index = MagicMock()
        mock_faiss_index.search.return_value = (np.array([[self.good_match_distance]]), self.match_id)
        
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (self.mock_historic_answer,)
        mock_db_conn = MagicMock()
        mock_db_conn.cursor.return_value = mock_cursor
        
        # This test doesn't use the mock_sqlite_connect directly for db_conn, 
        # as search_historic_qa receives an already connected db_conn.
        # The patch is more for completeness if other parts of engine used it globally.

        result = search_historic_qa(
            self.mock_question_embedding, mock_faiss_index, mock_db_conn, 
            k=1, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )

        self.assertEqual(result, self.mock_historic_answer)
        mock_faiss_index.search.assert_called_once_with(self.mock_question_embedding, 1)
        mock_db_conn.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with("SELECT answer_text FROM qa_pairs WHERE id = ?", (int(self.match_id[0][0]),))
        mock_cursor.fetchone.assert_called_once()


    def test_search_historic_qa_helper_no_match_faiss(self):
        """Test search_historic_qa helper directly: FAISS finds no match (distance too high)."""
        from engine import search_historic_qa

        mock_faiss_index = MagicMock()
        # Simulate FAISS returning a match, but its distance is too high
        mock_faiss_index.search.return_value = (np.array([[self.bad_match_distance]]), self.match_id)
        mock_db_conn = MagicMock()

        result = search_historic_qa(
            self.mock_question_embedding, mock_faiss_index, mock_db_conn, 
            k=1, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )
        self.assertIsNone(result)
        mock_faiss_index.search.assert_called_once_with(self.mock_question_embedding, 1)
        mock_db_conn.cursor.assert_not_called() # Should not query DB if distance too high

    def test_search_historic_qa_helper_no_match_faiss_empty_results(self):
        """Test search_historic_qa helper directly: FAISS returns empty results (e.g. k=0 or index empty)."""
        from engine import search_historic_qa

        mock_faiss_index = MagicMock()
        # Simulate FAISS returning no results (empty arrays for distances and IDs)
        mock_faiss_index.search.return_value = (np.array([[]]), np.array([[]]))
        mock_db_conn = MagicMock()

        result = search_historic_qa(
            self.mock_question_embedding, mock_faiss_index, mock_db_conn, 
            k=1, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )
        self.assertIsNone(result)
        mock_faiss_index.search.assert_called_once_with(self.mock_question_embedding, 1)
        mock_db_conn.cursor.assert_not_called()

    def test_search_historic_qa_helper_no_match_db(self):
        """Test search_historic_qa helper directly: FAISS finds match, but DB has no corresponding ID."""
        from engine import search_historic_qa

        mock_faiss_index = MagicMock()
        mock_faiss_index.search.return_value = (np.array([[self.good_match_distance]]), self.match_id)
        
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None # DB returns no row for the ID
        mock_db_conn = MagicMock()
        mock_db_conn.cursor.return_value = mock_cursor

        result = search_historic_qa(
            self.mock_question_embedding, mock_faiss_index, mock_db_conn, 
            k=1, max_distance_threshold=MAX_L2_DISTANCE_THRESHOLD
        )

        self.assertIsNone(result)
        mock_cursor.execute.assert_called_once_with("SELECT answer_text FROM qa_pairs WHERE id = ?", (int(self.match_id[0][0]),))


    @patch('engine.faiss') # Mock the faiss module used in engine
    @patch('engine.sqlite3') # Mock the sqlite3 module used in engine
    @patch('os.path.exists')
    def test_load_historic_resources_success(self, mock_os_path_exists, mock_sqlite3, mock_faiss):
        """Test load_historic_qa_resources successfully loads FAISS index and DB connection."""
        from engine import load_historic_qa_resources, HISTORIC_QA_DATA_DIR # Import locally
        
        grade = 11
        expected_index_path = os.path.join(HISTORIC_QA_DATA_DIR, f"grade{grade}.index")
        expected_db_path = os.path.join(HISTORIC_QA_DATA_DIR, f"grade{grade}_historic.db")

        # Configure os.path.exists to return True for these specific paths
        def side_effect_os_path_exists(path):
            if path == expected_index_path: return True
            if path == expected_db_path: return True
            return False
        mock_os_path_exists.side_effect = side_effect_os_path_exists

        mock_faiss_index_instance = MagicMock(name="FaissIndexInstance")
        mock_faiss.read_index.return_value = mock_faiss_index_instance
        
        mock_db_conn_instance = MagicMock(name="DbConnectionInstance")
        mock_sqlite3.connect.return_value = mock_db_conn_instance

        faiss_index, db_conn = load_historic_qa_resources(grade)

        self.assertEqual(faiss_index, mock_faiss_index_instance)
        self.assertEqual(db_conn, mock_db_conn_instance)
        
        mock_os_path_exists.assert_any_call(expected_index_path)
        mock_os_path_exists.assert_any_call(expected_db_path)
        mock_faiss.read_index.assert_called_once_with(expected_index_path)
        mock_sqlite3.connect.assert_called_once_with(expected_db_path)


    @patch('os.path.exists', return_value=False) # Simulate neither file existing
    def test_load_historic_resources_index_file_not_found(self, mock_os_path_exists):
        """Test load_historic_qa_resources returns (None, None) if FAISS index file not found."""
        from engine import load_historic_qa_resources
        faiss_index, db_conn = load_historic_qa_resources(11)
        self.assertIsNone(faiss_index)
        self.assertIsNone(db_conn) # If index fails, db connection is also aborted

    @patch('os.path.exists')
    def test_load_historic_resources_db_file_not_found(self, mock_os_path_exists):
        """Test load_historic_qa_resources returns (FaissIndex, None) if DB file not found but index is."""
        from engine import load_historic_qa_resources, HISTORIC_QA_DATA_DIR
        grade = 11
        expected_index_path = os.path.join(HISTORIC_QA_DATA_DIR, f"grade{grade}.index")
        expected_db_path = os.path.join(HISTORIC_QA_DATA_DIR, f"grade{grade}_historic.db")

        def side_effect_os_path_exists(path):
            if path == expected_index_path: return True # Index exists
            if path == expected_db_path: return False # DB does not exist
            return False
        mock_os_path_exists.side_effect = side_effect_os_path_exists
        
        # We still need to mock faiss.read_index if index path exists
        with patch('engine.faiss') as mock_faiss_module:
            mock_faiss_index_instance = MagicMock()
            mock_faiss_module.read_index.return_value = mock_faiss_index_instance
            
            faiss_index, db_conn = load_historic_qa_resources(grade)
            self.assertEqual(faiss_index, mock_faiss_index_instance) # Index loaded
            self.assertIsNone(db_conn) # DB connection failed
            mock_faiss_module.read_index.assert_called_once_with(expected_index_path)


if __name__ == '__main__':
    unittest.main()
