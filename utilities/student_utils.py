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
    """Extract information from student dictionary and convert grade to level string and numeric grade.
    
    Args:
        student_data (dict): The student's data dictionary
        
    Returns:
        tuple: (str, str, int): The vectorstore name, tablespace name, and numeric grade 
               (11 or 12, or 11 as default if not applicable).
    """

    vectorspace_mapping = {
        11: "grade-eleven-math",
        12: "grade-twelve-math"
    }

    tablespace_mapping = {
        11: "gradeElevenMath",
        12: "gradeTwelveMath"
    }

    try:
        student_id = student_data["Student ID"]
        logger.info(f"Getting student level for student {student_id}")
        
        # Extract grade from the student data dictionary
        grade = student_data["Current Class"]
        grade_int = int(grade)
        
        if grade_int == 12:
            logger.info(f"Student {student_id} is in grade 12")
            return vectorspace_mapping[12], tablespace_mapping[12], 12
        elif grade_int == 11:
            logger.info(f"Student {student_id} is in grade 11")
            return vectorspace_mapping[11], tablespace_mapping[11], 11
        else:
            logger.error(f"Student grade is {grade}, not 12th or 11th. Returning eleven as default")
            return vectorspace_mapping[11], tablespace_mapping[11], 11 
    except KeyError as e: 
        logger.error(f"Missing key in student data: {str(e)}")
        return vectorspace_mapping[11], tablespace_mapping[11], 11
    except Exception as e: 
        logger.error(f"Error getting student grade: {str(e)}")
        return vectorspace_mapping[11], tablespace_mapping[11], 11

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