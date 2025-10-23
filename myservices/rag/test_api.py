"""Integration tests for RAG search endpoint."""
import pytest
import requests
from typing import List, Dict

# Test configuration
BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
    {
        'query': 'global prevalence of diabetes',
        'expected_keywords': ['830 million', '2022', 'prevalence', 'low- and middle-income']
    },
    {
        'query': 'symptoms of diabetes',
        'expected_keywords': ['thirsty', 'urinate', 'blurred vision', 'tired', 'losing weight']
    },
    {
        'query': 'type 1 diabetes characteristics',
        'expected_keywords': ['insulin', 'deficient', 'daily', 'juvenile']
    },
    {
        'query': 'causes of type 2 diabetes',
        'expected_keywords': ['overweight', 'exercise', 'genetics', 'preventable']
    },
    {
        'query': 'gestational diabetes risks',
        'expected_keywords': ['pregnancy', 'delivery', 'complications', 'type 2']
    },
    {
        'query': 'prevention of type 2 diabetes',
        'expected_keywords': ['healthy diet', 'exercise', 'weight', 'tobacco']
    },
    {
        'query': 'treatment options for diabetes',
        'expected_keywords': ['insulin', 'metformin', 'sulfonylureas', 'sglt-2']
    },
    {
        'query': 'diabetes complications',
        'expected_keywords': ['blindness', 'kidney', 'heart', 'stroke', 'amputation']
    },
    {
        'query': 'WHO global diabetes compact',
        'expected_keywords': ['global diabetes compact', '2021', 'initiative', 'prevention']
    },
    {
        'query': 'diabetes mortality statistics',
        'expected_keywords': ['1.6 million', 'deaths', '70 years', 'cardiovascular']
    }
]


def check_response_structure(response: Dict) -> bool:
    """Verify the response has the expected structure."""
    return (
        isinstance(response, dict) 
        and 'query' in response 
        and 'results' in response
        and isinstance(response['results'], list)
        and len(response['results']) > 0
        and all(
            isinstance(result, dict) 
            and all(key in result for key in ['doc_id', 'text', 'score'])
            for result in response['results']
        )
    )

def contains_keywords(text: str, keywords: List[str]) -> bool:
    """Check if text contains all expected keywords."""
    text = text.lower()
    return all(keyword.lower() in text for keyword in keywords)

def test_search_endpoint():
    """Test search endpoint with multiple queries."""
    successful_queries = 0
    
    for test_case in TEST_QUERIES:
        # Make request
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                'query': test_case['query'],
                'top_k': 3
            }
        )
        
        # Check response status
        assert response.status_code == 200, f"Query failed: {test_case['query']}"
        
        # Parse response
        data = response.json()
        
        # Verify response structure
        assert check_response_structure(data), "Invalid response structure"
        
        # Check if any result contains all expected keywords
        found_keywords = any(
            contains_keywords(result['text'], test_case['expected_keywords'])
            for result in data['results']
        )
        
        if found_keywords:
            successful_queries += 1
    
    # Ensure at least 80% of queries were successful
    success_rate = successful_queries / len(TEST_QUERIES)
    assert success_rate >= 0.8, f"Success rate too low: {success_rate*100:.1f}%"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
