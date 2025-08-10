import pandas as pd
import os
from logger_config import setup_logger

logger = setup_logger(__name__)

# Determine the project root (assuming student_utils.py is in utilities/ subdir)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, "official_db.csv")

CONFIDENCE_LEVELS_MAPPING = {
    "1= = Not confident": 1,
    "2 = Slightly confident": 2,
    "3 = Neutral": 3,
    "4 = Confident": 4,
    "5 = Very confident": 5
}

SCORES_MAPPING = {
    "A (64-71)": "A",
    "A+ (72–80)": "A+",
    "B (48-55)": "B",
    "B+ (56-63)": "B+",
    "C (40-47)": "C",
    "D (32-39)": "D",
}

def get_student_level(student_data):
    """Extract information from student dictionary and return grade 12 configuration.
    
    Args:
        student_data (dict): The student's data dictionary
        
    Returns:
        tuple: (str, str, int): The vectorstore name, tablespace name, and numeric grade (always 12).
    """

    # Only supporting grade 12 students
    vectorstore_name = "grade-twelve-math"
    tablespace_name = "gradeTwelveMath"
    grade = 12

    try:
        student_id = student_data["Student ID"]
        logger.info(f"Getting student level for student {student_id} (Grade 12 only)")
        
        # Verify the student is in grade 12 (optional validation)
        current_class = student_data.get("Current Class", "12")
        if str(current_class) != "12":
            logger.warning(f"Student {student_id} is in grade {current_class}, but system only supports grade 12. Proceeding with grade 12 configuration.")
        
        logger.info(f"Student {student_id} configured for grade 12")
        return vectorstore_name, tablespace_name, grade
        
    except KeyError as e: 
        logger.error(f"Missing key in student data: {str(e)}. Using grade 12 defaults.")
        return vectorstore_name, tablespace_name, grade
    except Exception as e: 
        logger.error(f"Error getting student grade: {str(e)}. Using grade 12 defaults.")
        return vectorstore_name, tablespace_name, grade

def get_confidence_level_and_score(student_data, topic):
    """Get student's confidence level (1-5) and score for a specific topic from student data dictionary.
    
    Args:
        student_data (dict): The student's data dictionary
        topic (str): The topic to get the confidence level for
    Returns:
        tuple: (int, str): The student's confidence level as a integer (1-5) and score as a string ('A', 'B', etc.), defaults to (1, '') on error.
    """
    try:
        student_id = student_data['Student ID']
        logger.info(f"Getting confidence level for student {student_id} in topic {topic}")
        
        # Build a mapping of lowercase column names to their original forms for case-insensitive lookup
        topic_columns_map = {key.lower(): key for key in student_data.keys()}
        topic_lower = topic.lower()

        if topic_lower not in topic_columns_map:
            logger.warning(
                f"Topic '{topic}' not found as a key (case-insensitive) in student data. "
                f"Available keys: {list(student_data.keys())}. Returning default level 1."
            )
            return 1, ''

        # Get the actual key name (with correct case)
        topic_key_actual = topic_columns_map[topic_lower]
        confidence_level_raw = student_data[topic_key_actual]
        score_raw = student_data.get("Score", "")

        # Parse confidence level and score
        confidence_level_num = CONFIDENCE_LEVELS_MAPPING.get(confidence_level_raw, 1)
        score = SCORES_MAPPING.get(score_raw, "")
        
        logger.info(f"Student {student_id} is at {confidence_level_num} level and {score} score for {topic}")
        return int(confidence_level_num), score
        
    except KeyError as e:
        logger.error(f"KeyError accessing student data for topic {topic}: {e}. Missing key in student data dictionary.")
        return 1, ''
    except Exception as e:
        logger.error(f"Error getting student topic level for topic {topic}: {str(e)}")
        return 1, ''

def check_confidence_and_score(confidence_level, score):
    if confidence_level == 4 or confidence_level == 5:
        if score == "A" or score == "A+":
            return True
        else:
            return False
    else:
        return False