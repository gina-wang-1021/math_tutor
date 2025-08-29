import pandas as pd
import os
from logger_config import setup_logger

logger = setup_logger(__name__)

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

def get_confidence_level_and_score(score_field, student_data, topic):
    """Get student's confidence level (1-5) and score for a specific topic from student data dictionary.
    
    Args:
        student_data (dict): The student's data dictionary
        topic (str): The topic to get the confidence level for
    Returns:
        tuple: (int, str): The student's confidence level as a integer (1-5) and score as a string ('A', 'B', etc.), defaults to (1, '') on error.
    """
    
    econ_topic_mapping = {
        "intro": "introduction to economics",
        "consumer": "consumer behavior and demand",
        "producer": "producer behavior and supply",
        "equilibrium": "market equilibrium",
        "data": "collection, organisation and presentation of data",
        "stats": "statistical tools and interpretation"
    }
    
    try:
        username = student_data['Username']
        logger.info(f"Getting confidence level for student {username} in topic {topic}")
        
        # Build a mapping of lowercase column names to their original forms for case-insensitive lookup
        topic_columns_map = {key.lower(): key for key in student_data.keys()}
        topic_lower = topic.lower()
        if topic_lower in econ_topic_mapping:
            topic = econ_topic_mapping[topic_lower]

        if topic not in topic_columns_map:
            logger.warning(
                f"Topic '{topic}' not found as a key (case-insensitive) in student data. "
                f"Available keys: {list(student_data.keys())}. Returning default level 1."
            )
            return 1, ''

        # Get the actual key name (with correct case)
        topic_key_actual = topic_columns_map[topic]
        confidence_level_raw = student_data[topic_key_actual]
        score_raw = student_data.get(score_field, "")

        # Parse confidence level and score
        confidence_level_num = CONFIDENCE_LEVELS_MAPPING.get(confidence_level_raw, 1)
        score = SCORES_MAPPING.get(score_raw, "")
        
        logger.info(f"Student {username} is at {confidence_level_num} level and {score} score for {topic}")
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