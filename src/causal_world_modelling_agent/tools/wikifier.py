
from typing import Optional
import requests

from smolagents import tool
from ..core.definitions import Message

@tool
def findCorrespondingWikiDataConcept(variable_name: str) -> Optional[Message]:
    """
    Call the WikiData knowledge graph API to find the corresponding concept in the knowledge graph for the given variable name.
    Args:
        variable_name: The name of the variable to find in the knowledge graph.
    Returns:
        A Message object containing the WikiData concept information if found, otherwise None.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": variable_name
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['search']:
            concept = data['search'][0]
            return {
                'id': concept['id'],
                'label': concept['label'],
                'description': concept.get('description', ''),
                'url': f"https://www.wikidata.org/wiki/{concept['id']}"
            }
    return None