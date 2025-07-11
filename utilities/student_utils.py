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

def get_student_level(student_id):
    """Get student's grade from student_data.csv and convert grade to level string and numeric grade.
    
    Args:
        student_id (str): The student's ID to look up
        
    Returns:
        tuple: (str, int or None): The database name of that grade 
               and numeric grade (11 or 12, or None if not applicable for historic Q&A).
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
        if not os.path.exists(STUDENT_DATA_CSV_PATH):
            logger.error(f"official_db.csv not found at {STUDENT_DATA_CSV_PATH}")
            return None, None, None
        df = pd.read_csv(STUDENT_DATA_CSV_PATH)
        # Convert both the column and the parameter to string for comparison
        student_data = df[df["Student ID"].astype(str) == str(student_id)]
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database ({STUDENT_DATA_CSV_PATH})")
            return None, None, None
        
        grade = student_data.iloc[0]["Current Class"]
        if int(grade) == 12:
            return vectorspace_mapping[12], tablespace_mapping[12], 12
        else:
            logger.error(f"Student grade is {grade}, not 12th or 11th. Returning eleven as default")
            return vectorspace_mapping[11], tablespace_mapping[11], 11 
    except Exception as e: 
        logger.error(f"Error getting student grade for {student_id}: {str(e)}")
        return None, None, None

def get_confidence_level_and_score(student_id, topic):
    """Get student's confidence level (1-5) and score for a specific topic from student_data.csv.
    
    Args:
        student_id (str): The student's ID to look up
        topic (str): The topic to get the confidence level for
    Returns:
        tuple: (int, str): The student's confidence level as a integer (1-5) and score as a string ('A', 'B', etc.), defaults to (1, '') on error.
    """
    try:
        logger.info(f"Getting confidence level for student {student_id} in topic {topic}")
        
        if not os.path.exists(STUDENT_DATA_CSV_PATH):
            logger.error(f"official_db.csv not found at {STUDENT_DATA_CSV_PATH}")
            return 1, ''
            
        df = pd.read_csv(STUDENT_DATA_CSV_PATH)
        student_data = df[df["Student ID"].astype(str) == str(student_id)]
        
        if student_data.empty:
            logger.warning(f"Student {student_id} not found in database ({STUDENT_DATA_CSV_PATH})")
            return 1, ''
        
        # Build a mapping of lowercase column names to their original forms for case-insensitive lookup
        topic_columns_map = {col.lower(): col for col in student_data.columns}

        if topic not in topic_columns_map:
            logger.warning(
                f"Topic '{topic}' not found as a column (case-insensitive) in {STUDENT_DATA_CSV_PATH}. "
                f"Student data columns: {student_data.columns.tolist()}. Returning default level 1."
            )
            return 1, ''

        topic_column_actual = topic_columns_map[topic]
        confidence_level_raw = student_data.iloc[0][topic_column_actual]
        score_raw = student_data.iloc[0]["Score"]

        confidence_level_num = CONFIDENCE_LEVELS_MAPPING.get(confidence_level_raw, 1)
        score = SCORES_MAPPING.get(score_raw, "")
        
        logger.info(f"Student {student_id} is at {confidence_level_num} level and {score} score for {topic}")
        return int(confidence_level_num), score
        
    except pd.errors.EmptyDataError:
        logger.error(f"official_db.csv at {STUDENT_DATA_CSV_PATH} is empty or malformed.")
        return 1, ''
    except KeyError as e:
        logger.error(f"KeyError accessing student data for {student_id}, topic {topic}: {e}. This might mean student_id or topic column is missing or misspelled in {STUDENT_DATA_CSV_PATH}.")
        return 1, ''
    except Exception as e:
        logger.error(f"Error getting student topic level for {student_id}, topic {topic}: {str(e)}")
        return 1, ''

def check_confidence_and_score(confidence_level, score):
    if confidence_level == 4 or confidence_level == 5:
        if score == "A" or score == "A+":
            return True
        else:
            return False
    else:
        return False