import pandas as pd
import os
from logger_config import setup_logger

logger = setup_logger(__name__)

# Determine the project root (assuming student_utils.py is in utilities/ subdir)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "student_data.csv")

def get_student_level(student_id):
    """Get student's grade from student_data.csv and convert grade to level string and numeric grade.
    
    Args:
        student_id (str): The student's ID to look up
        
    Returns:
        tuple: (str, int or None): The student's level string (e.g., 'intermediate') 
               and numeric grade (11 or 12, or None if not applicable for historic Q&A).
    """
    try:
        if not os.path.exists(STUDENT_DATA_CSV_PATH):
            logger.error(f"student_data.csv not found at {STUDENT_DATA_CSV_PATH}")
            return None, None
        df = pd.read_csv(STUDENT_DATA_CSV_PATH)
        student_data = df[df["student_id"] == student_id]
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database ({STUDENT_DATA_CSV_PATH})")
            return None, None
        
        grade = student_data.iloc[0]["grade"]
        if grade == 11:
            return "intermediate", 11
        elif grade == 12:
            return "advanced", 12
        else:
            # For grades other than 11/12, or if grade is not a number that can be 11/12
            level = "beginner" # Default level for other grades
            return level, None 
    except Exception as e:
        logger.error(f"Error getting student grade for {student_id}: {str(e)}")
        return None, None

def get_student_topic_level(student_id, topic):
    """Get student's level (0 to 5) for a specific topic from student_data.csv.
    
    Args:
        student_id (str): The student's ID to look up
        topic (str): The topic to get the level for
        
    Returns:
        int: The student's level as an integer from 0 to 5, defaults to 0 on error.
    """
    try:
        logger.info(f"Getting level for student {student_id} in topic {topic}")
        
        if not os.path.exists(STUDENT_DATA_CSV_PATH):
            logger.error(f"student_data.csv not found at {STUDENT_DATA_CSV_PATH}")
            return 0
            
        df = pd.read_csv(STUDENT_DATA_CSV_PATH)
        student_data = df[df["student_id"] == student_id]
        
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database ({STUDENT_DATA_CSV_PATH})")
            return 0
        
        if topic not in student_data.columns:
            logger.warning(f"Topic '{topic}' not found as a column in {STUDENT_DATA_CSV_PATH}. Student data columns: {student_data.columns.tolist()}. Returning default level 0.")
            return 0
            
        student_level = student_data.iloc[0][topic]
        logger.info(f"Student {student_id} is at {student_level} level for {topic}")
        return int(student_level) # Ensure it's an int
        
    except pd.errors.EmptyDataError:
        logger.error(f"student_data.csv at {STUDENT_DATA_CSV_PATH} is empty or malformed.")
        return 0
    except KeyError as e:
        logger.error(f"KeyError accessing student data for {student_id}, topic {topic}: {e}. This might mean student_id or topic column is missing or misspelled in {STUDENT_DATA_CSV_PATH}.")
        return 0
    except Exception as e:
        logger.error(f"Error getting student topic level for {student_id}, topic {topic}: {str(e)}")
        return 0
