# nltk_helpers.py - Wrappers for NLTK functionality
import os
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import random
from pathlib import Path

logger = logging.getLogger("NLTKHelpers")

# Dictionary to track which resources are already downloaded
downloaded_resources = set()

def ensure_nltk_resources(resources: List[str]) -> bool:
    """Ensure required NLTK resources are downloaded
    
    Args:
        resources: List of NLTK resource names to download
    
    Returns:
        True if all resources were successfully downloaded/available
    """
    global downloaded_resources
    
    # Skip already downloaded resources
    resources_to_download = [r for r in resources if r not in downloaded_resources]
    
    if not resources_to_download:
        return True
    
    try:
        import nltk
        
        # Set custom download directory if not running in regular environment
        nltk_data_path = os.environ.get('NLTK_DATA', str(Path.home() / 'nltk_data'))
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk.data.path.append(nltk_data_path)
        
        # Download resources
        for resource in resources_to_download:
            try:
                nltk.download(resource, quiet=True, download_dir=nltk_data_path)
                downloaded_resources.add(resource)
                logger.info(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Could not download NLTK resource {resource}: {str(e)}")
                return False
        
        return True
    
    except ImportError:
        logger.warning("NLTK not available, some features will be disabled")
        return False

def get_phonetic_representation(word: str) -> str:
    """Get phonetic representation of a word using CMU pronouncing dictionary
    
    Args:
        word: The word to get phonetic representation for
    
    Returns:
        Phonetic representation or empty string if not available
    """
    try:
        if not ensure_nltk_resources(['cmudict']):
            return ""
        
        import nltk
        from nltk.corpus import cmudict
        
        try:
            # Get pronunciation
            pronunciation_dict = cmudict.dict()
            if word.lower() in pronunciation_dict:
                # Get first pronunciation
                phonemes = pronunciation_dict[word.lower()][0]
                # Remove stress markers
                phonemes = [p.strip("0123456789") for p in phonemes]
                return " ".join(phonemes)
        except Exception as e:
            logger.warning(f"Error getting phonetic representation for '{word}': {str(e)}")
        
        return ""
    
    except ImportError:
        logger.warning("NLTK or cmudict not available, phonetic features disabled")
        return ""

def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text using NLTK for parts of speech, entities, etc.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary of analysis results
    """
    try:
        if not ensure_nltk_resources(['punkt', 'averaged_perceptron_tagger']):
            return _analyze_text_fallback(text)
        
        import nltk
        
        # Tokenize and tag
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        # Count parts of speech
        pos_counts = {}
        for word, tag in tagged:
            if tag not in pos_counts:
                pos_counts[tag] = 0
            pos_counts[tag] += 1
        
        # Extract parts of speech
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        verbs = [word for word, tag in tagged if tag.startswith('VB')]
        adjectives = [word for word, tag in tagged if tag.startswith('JJ')]
        adverbs = [word for word, tag in tagged if tag.startswith('RB')]
        
        # Simple sentence structure analysis
        sentence_structure = "unknown"
        if len(tokens) == 1:
            sentence_structure = "single_word"
        elif len(tokens) <= 3:
            sentence_structure = "telegraphic"
        elif len(tagged) > 3:
            # Look for subject-verb-object pattern
            has_noun = any(tag.startswith('NN') for _, tag in tagged)
            has_verb = any(tag.startswith('VB') for _, tag in tagged)
            if has_noun and has_verb:
                sentence_structure = "subject_verb"
        
        return {
            "tokens": tokens,
            "tagged": tagged,
            "pos_counts": pos_counts,
            "nouns": nouns,
            "verbs": verbs,
            "adjectives": adjectives,
            "adverbs": adverbs,
            "sentence_structure": sentence_structure,
            "word_count": len(tokens),
            "avg_word_length": sum(len(word) for word in tokens) / max(1, len(tokens))
        }
    
    except ImportError:
        logger.warning("NLTK not available, using fallback text analysis")
        return _analyze_text_fallback(text)

def _analyze_text_fallback(text: str) -> Dict[str, Any]:
    """Fallback text analysis for when NLTK is not available
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary of analysis results
    """
    # Simple tokenization
    tokens = text.split()
    
    # Very basic POS guessing
    tagged = []
    for word in tokens:
        # Guess part of speech based on simple rules
        if word.lower() in ["a", "an", "the"]:
            tag = "DT"  # Determiner
        elif word.lower() in ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]:
            tag = "PRP"  # Personal pronoun
        elif word.lower() in ["is", "am", "are", "was", "were", "be", "been", "being"]:
            tag = "VB"  # Verb
        elif word.lower().endswith("ly"):
            tag = "RB"  # Adverb
        elif word.lower().endswith(("ed", "ing")):
            tag = "VBG"  # Verb
        else:
            tag = "NN"  # Default to noun
        
        tagged.append((word, tag))
    
    # Count tags
    pos_counts = {}
    for _, tag in tagged:
        if tag not in pos_counts:
            pos_counts[tag] = 0
        pos_counts[tag] += 1
    
    # Extract parts of speech
    nouns = [word for word, tag in tagged if tag == "NN"]
    verbs = [word for word, tag in tagged if tag == "VB" or tag == "VBG"]
    adjectives = [word for word, tag in tagged if tag == "JJ"]
    adverbs = [word for word, tag in tagged if tag == "RB"]
    
    # Simple sentence structure
    sentence_structure = "unknown"
    if len(tokens) == 1:
        sentence_structure = "single_word"
    elif len(tokens) <= 3:
        sentence_structure = "telegraphic"
    else:
        sentence_structure = "multi_word"
    
    return {
        "tokens": tokens,
        "tagged": tagged,
        "pos_counts": pos_counts,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjectives,
        "adverbs": adverbs,
        "sentence_structure": sentence_structure,
        "word_count": len(tokens),
        "avg_word_length": sum(len(word) for word in tokens) / max(1, len(tokens))
    }

def simplify_text(text: str, vocabulary_size: int) -> str:
    """Simplify text based on vocabulary size
    
    Args:
        text: Text to simplify
        vocabulary_size: Target vocabulary size
    
    Returns:
        Simplified text
    """
    try:
        if not ensure_nltk_resources(['punkt', 'averaged_perceptron_tagger', 'stopwords']):
            return text
        
        import nltk
        from nltk.corpus import stopwords
        
        # For very small vocabularies, just return a greatly simplified version
        if vocabulary_size < 50:
            words = nltk.word_tokenize(text)
            # Keep only first 5-7 words
            limit = min(7, len(words))
            return " ".join(words[:limit])
        
        # For small vocabularies, remove complex words and shorten
        elif vocabulary_size < 200:
            words = nltk.word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            
            # Keep simple words, determiners, and pronouns
            simple_words = []
            for word in words:
                # Keep words if they're stopwords, short, or appear in first part of the text
                if (word.lower() in stop_words or 
                    len(word) <= 4 or 
                    len(simple_words) < 5):
                    simple_words.append(word)
            
            return " ".join(simple_words)
        
        # For medium vocabularies, keep sentence structure but simplify
        elif vocabulary_size < 500:
            sentences = nltk.sent_tokenize(text)
            result = []
            
            # Take only first 1-2 sentences
            for i, sentence in enumerate(sentences[:2]):
                words = nltk.word_tokenize(sentence)
                # Simplify long sentences
                if len(words) > 8:
                    words = words[:8]
                result.append(" ".join(words))
            
            return " ".join(result)
        
        # For larger vocabularies, minimal simplification
        else:
            sentences = nltk.sent_tokenize(text)
            result = []
            
            # Take only first 3 sentences
            for sentence in sentences[:3]:
                result.append(sentence)
            
            return " ".join(result)
    
    except ImportError:
        logger.warning("NLTK not available, text simplification limited")
        # Simple fallback - just truncate
        words = text.split()
        max_words = min(10, len(words))
        return " ".join(words[:max_words])

def generate_child_speech_errors(text: str, age_days: float) -> str:
    """Generate age-appropriate speech errors
    
    Args:
        text: Text to apply errors to
        age_days: Child's age in days
    
    Returns:
        Text with age-appropriate errors
    """
    # No errors for empty text
    if not text:
        return text
    
    words = text.split()
    result = []
    
    # Determine error probabilities based on age
    if age_days < 10:  # Very early stage
        # High probability of errors
        phoneme_sub_prob = 0.7
        word_trunc_prob = 0.6
        word_drop_prob = 0.5
    elif age_days < 30:  # Early stage
        phoneme_sub_prob = 0.5
        word_trunc_prob = 0.4
        word_drop_prob = 0.3
    elif age_days < 100:  # Middle stage
        phoneme_sub_prob = 0.3
        word_trunc_prob = 0.2
        word_drop_prob = 0.1
    else:  # Later stage
        phoneme_sub_prob = 0.1
        word_trunc_prob = 0.05
        word_drop_prob = 0.02
    
    # Common phoneme substitutions for young children
    phoneme_subs = {
        'r': 'w',
        'l': 'w',
        'th': 'd',
        'f': 'p',
        'v': 'b',
        'z': 's',
        'sh': 's',
        'ch': 't'
    }
    
    # Apply errors to each word
    for word in words:
        # Original word with punctuation preserved
        original = word
        # Clean word for processing
        clean_word = word.strip(".,;:!?\"'()[]{}").lower()
        
        # Skip very short words
        if len(clean_word) <= 2:
            result.append(original)
            continue
        
        # Word dropping
        if random.random() < word_drop_prob:
            # Skip function words
            if clean_word not in ["the", "a", "an", "in", "on", "at", "to", "for", "of"]:
                continue
        
        # Word truncation
        if len(clean_word) > 3 and random.random() < word_trunc_prob:
            if random.random() < 0.5:
                # Keep first part of word
                keep_length = random.randint(1, len(clean_word) - 1)
                clean_word = clean_word[:keep_length]
            else:
                # Simplify consonant clusters
                for cluster in ["str", "pl", "bl", "gr", "dr", "tr", "fr", "th"]:
                    if cluster in clean_word:
                        replacement = cluster[0]
                        clean_word = clean_word.replace(cluster, replacement)
        
        # Phoneme substitutions
        if random.random() < phoneme_sub_prob:
            for phoneme, replacement in phoneme_subs.items():
                if phoneme in clean_word:
                    clean_word = clean_word.replace(phoneme, replacement)
                    break  # Only do one substitution per word
        
        # Preserve punctuation
        for c in original:
            if not c.isalnum():
                clean_word += c
        
        result.append(clean_word)
    
    # Return final text with errors
    return " ".join(result)

def check_child_syntax(child_text: str, expected_stage: str) -> Dict[str, Any]:
    """Check if child's syntax matches expected developmental stage
    
    Args:
        child_text: Child's speech
        expected_stage: Expected developmental stage
    
    Returns:
        Dictionary with analysis results
    """
    analysis = analyze_text(child_text)
    
    # Determine expected features by stage
    if expected_stage == "pre_linguistic":
        expected_features = {
            "word_count": (0, 0),  # (min, max)
            "sentence_structure": "single_word",
            "has_verbs": False,
            "has_determiners": False
        }
    elif expected_stage == "holophrastic":
        expected_features = {
            "word_count": (0, 1),
            "sentence_structure": "single_word",
            "has_verbs": False,
            "has_determiners": False
        }
    elif expected_stage == "telegraphic":
        expected_features = {
            "word_count": (1, 3),
            "sentence_structure": "telegraphic",
            "has_verbs": True,
            "has_determiners": False
        }
    elif expected_stage == "simple_syntax":
        expected_features = {
            "word_count": (2, 5),
            "sentence_structure": "subject_verb",
            "has_verbs": True,
            "has_determiners": True
        }
    else:  # complex_syntax or advanced
        expected_features = {
            "word_count": (3, 10),
            "sentence_structure": "subject_verb",
            "has_verbs": True,
            "has_determiners": True
        }
    
    # Check actual features
    actual_features = {
        "word_count": analysis["word_count"],
        "sentence_structure": analysis["sentence_structure"],
        "has_verbs": len(analysis["verbs"]) > 0,
        "has_determiners": any(tag == "DT" for _, tag in analysis.get("tagged", []))
    }
    
    # Check if features match
    matches = {
        "word_count": expected_features["word_count"][0] <= actual_features["word_count"] <= expected_features["word_count"][1],
        "sentence_structure": actual_features["sentence_structure"] == expected_features["sentence_structure"],
        "has_verbs": actual_features["has_verbs"] == expected_features["has_verbs"],
        "has_determiners": actual_features["has_determiners"] == expected_features["has_determiners"]
    }
    
    # Calculate overall match
    match_count = sum(1 for match in matches.values() if match)
    overall_match = match_count / len(matches)
    
    return {
        "expected_features": expected_features,
        "actual_features": actual_features,
        "matches": matches,
        "overall_match": overall_match
    }