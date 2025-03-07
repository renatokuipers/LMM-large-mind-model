"""
Test script for NLP utilities in the semantic memory extraction system.
"""
from utils.nlp_utils import (
    extract_named_entities,
    extract_key_concepts,
    extract_relationships,
    categorize_knowledge
)
from nltk.tokenize import sent_tokenize

def main():
    """Run tests for NLP utilities."""
    # Sample text for testing
    test_text = """
    John Smith, the CEO of Acme Corporation, announced a new AI product on January 15th, 2023.
    The product uses advanced machine learning algorithms to improve customer service.
    According to Smith, this technology will revolutionize how companies interact with their customers
    because it can understand complex queries and respond with high accuracy.
    The company's stock price increased by 5% after the announcement.
    """
    
    print("Testing NLP utilities for semantic memory extraction")
    print("-" * 70)
    
    # Step 1: Extract named entities
    print("\n1. Named Entity Extraction:")
    entities = extract_named_entities(test_text)
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type}: {', '.join(entity_list)}")
    
    # Step 2: Extract key concepts
    print("\n2. Key Concept Extraction:")
    concepts = extract_key_concepts(test_text)
    for concept in concepts[:5]:  # Show top 5 concepts
        print(f"  {concept['term']} ({concept['type']}): importance={concept['importance']:.2f}")
    
    # Step 3: Extract relationships
    print("\n3. Relationship Extraction:")
    sentences = sent_tokenize(test_text)
    relationships = extract_relationships(sentences, entities, concepts)
    for rel in relationships[:3]:  # Show first 3 relationships
        print(f"  {rel['source']} -> {rel['target']} ({rel['relation_type']})")
    
    # Step 4: Knowledge categorization
    print("\n4. Knowledge Categorization:")
    categorized = categorize_knowledge(test_text, concepts, entities, relationships)
    for item in categorized[:3]:  # Show first 3 categorized items
        print(f"  {item['category']}/{item['subcategory']} ({item['confidence']:.2f}): {item['content'][:70]}...")

if __name__ == "__main__":
    main() 