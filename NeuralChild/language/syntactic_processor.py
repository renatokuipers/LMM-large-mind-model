# syntactic_processor.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
from pydantic import BaseModel, Field, field_validator
from enum import Enum, auto
import numpy as np

from language.developmental_stages import LanguageDevelopmentStage, LanguageCapabilities, LanguageFeature

logger = logging.getLogger("SyntacticProcessor")

class GrammaticalCategory(str, Enum):
    """Grammatical categories that develop over time"""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    DETERMINER = "determiner"
    PREPOSITION = "preposition"
    PRONOUN = "pronoun"
    CONJUNCTION = "conjunction"
    AUXILIARY = "auxiliary"  # Auxiliary verbs like "is", "have", "can"
    TENSE = "tense"          # Past, present, future
    NUMBER = "number"        # Singular, plural
    NEGATION = "negation"    # Negative forms

class GrammaticalRule(BaseModel):
    """A grammatical rule that can be applied in language production"""
    name: str
    description: str
    min_stage: LanguageDevelopmentStage
    categories: List[GrammaticalCategory]
    examples: List[str]
    acquisition_difficulty: float = Field(0.5, ge=0.0, le=1.0)
    mastery_level: float = Field(0.0, ge=0.0, le=1.0)
    first_used: Optional[datetime] = None
    usage_count: int = Field(0, ge=0)
    
    def update_mastery(self, success: bool, learning_rate: float = 0.05) -> None:
        """Update mastery level based on usage success"""
        if not self.first_used:
            self.first_used = datetime.now()
        
        self.usage_count += 1
        
        if success:
            # Successful use increases mastery (with diminishing returns)
            increment = learning_rate * (1.0 - self.mastery_level)
            self.mastery_level = min(1.0, self.mastery_level + increment)
        else:
            # Unsuccessful use slightly decreases mastery
            self.mastery_level = max(0.0, self.mastery_level - (learning_rate * 0.5))

class SyntacticProcessor:
    """Manages syntactic processing and grammar development"""
    
    def __init__(self):
        """Initialize syntactic processor"""
        # Try to load spaCy for advanced processing
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            logger.info("spaCy loaded successfully for syntactic processing")
        except (ImportError, OSError):
            logger.warning("spaCy not available; using simplified syntactic processing")
            self.nlp = None
            self.spacy_available = False
        
        # Initialize set of grammatical rules
        self.grammar_rules = self._initialize_grammar_rules()
        
        # Track grammar error types and frequencies for realistic child language
        self.error_patterns = {
            LanguageDevelopmentStage.PRE_LINGUISTIC: {},
            LanguageDevelopmentStage.HOLOPHRASTIC: {
                "word_omission": 0.9,       # Omitting words
                "phonetic_simplification": 0.8  # Simplifying word pronunciation
            },
            LanguageDevelopmentStage.TELEGRAPHIC: {
                "word_omission": 0.7,       # Omitting words, especially function words
                "word_order": 0.5,          # Incorrect word order
                "inflection_omission": 0.8,  # Missing -s, -ed, -ing
                "determiner_omission": 0.9,  # Missing a, the
                "preposition_omission": 0.8  # Missing in, on, etc.
            },
            LanguageDevelopmentStage.SIMPLE_SYNTAX: {
                "word_omission": 0.4,
                "word_order": 0.3,
                "inflection_omission": 0.5,
                "determiner_omission": 0.4,
                "preposition_omission": 0.5,
                "pronoun_error": 0.6,        # me/I confusion
                "tense_error": 0.7,          # Incorrect verb tense
                "agreement_error": 0.6       # Subject-verb agreement issues
            },
            LanguageDevelopmentStage.COMPLEX_SYNTAX: {
                "word_omission": 0.2,
                "word_order": 0.1,
                "inflection_omission": 0.3,
                "determiner_omission": 0.2,
                "preposition_omission": 0.3,
                "pronoun_error": 0.3,
                "tense_error": 0.4,
                "agreement_error": 0.4,
                "complex_construction": 0.6   # Errors in relative clauses, etc.
            },
            LanguageDevelopmentStage.ADVANCED: {
                "word_omission": 0.05,
                "word_order": 0.05,
                "inflection_omission": 0.1,
                "determiner_omission": 0.05,
                "preposition_omission": 0.1,
                "pronoun_error": 0.1,
                "tense_error": 0.1,
                "agreement_error": 0.1,
                "complex_construction": 0.3
            }
        }
        
        logger.info("Syntactic processor initialized")
    
    def _initialize_grammar_rules(self) -> List[GrammaticalRule]:
        """Initialize the set of grammatical rules that develop over time"""
        rules = []
        
        # Basic rules
        rules.append(GrammaticalRule(
            name="single_word_naming",
            description="Single words used to name objects or actions",
            min_stage=LanguageDevelopmentStage.HOLOPHRASTIC,
            categories=[GrammaticalCategory.NOUN, GrammaticalCategory.VERB],
            examples=["ball", "milk", "more"],
            acquisition_difficulty=0.1
        ))
        
        rules.append(GrammaticalRule(
            name="agent_action",
            description="Agent + action with no grammatical markers (telegraphic)",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            categories=[GrammaticalCategory.NOUN, GrammaticalCategory.VERB],
            examples=["Mommy go", "Daddy eat", "Baby sleep"],
            acquisition_difficulty=0.2
        ))
        
        rules.append(GrammaticalRule(
            name="agent_action_object",
            description="Agent + action + object with no grammatical markers",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            categories=[GrammaticalCategory.NOUN, GrammaticalCategory.VERB],
            examples=["Daddy throw ball", "Baby want milk"],
            acquisition_difficulty=0.3
        ))
        
        rules.append(GrammaticalRule(
            name="possession",
            description="Expressing possession with 'my' or name + possession",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            categories=[GrammaticalCategory.PRONOUN, GrammaticalCategory.NOUN],
            examples=["my ball", "Daddy car"],
            acquisition_difficulty=0.3
        ))
        
        rules.append(GrammaticalRule(
            name="negative",
            description="Simple negation with 'no' or 'not'",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            categories=[GrammaticalCategory.NEGATION],
            examples=["no milk", "not bed", "no go"],
            acquisition_difficulty=0.3
        ))
        
        # Simple syntax rules
        rules.append(GrammaticalRule(
            name="subject_verb_agreement",
            description="Agreement between subject and verb",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.NUMBER],
            examples=["I am", "She is", "They are"],
            acquisition_difficulty=0.5
        ))
        
        rules.append(GrammaticalRule(
            name="article_usage",
            description="Using articles 'a' and 'the' appropriately",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            categories=[GrammaticalCategory.DETERMINER],
            examples=["the ball", "a dog"],
            acquisition_difficulty=0.4
        ))
        
        rules.append(GrammaticalRule(
            name="present_continuous",
            description="Using present continuous tense with -ing",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.TENSE, GrammaticalCategory.AUXILIARY],
            examples=["I am playing", "He is eating"],
            acquisition_difficulty=0.5
        ))
        
        rules.append(GrammaticalRule(
            name="preposition_usage",
            description="Using basic prepositions like in, on, under",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            categories=[GrammaticalCategory.PREPOSITION],
            examples=["in box", "on table", "under chair"],
            acquisition_difficulty=0.4
        ))
        
        rules.append(GrammaticalRule(
            name="simple_question",
            description="Forming simple questions",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.AUXILIARY],
            examples=["Where ball?", "What that?", "Can I go?"],
            acquisition_difficulty=0.5
        ))
        
        # Complex syntax rules
        rules.append(GrammaticalRule(
            name="past_tense",
            description="Using regular and irregular past tense",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.TENSE],
            examples=["I played", "He went", "She ate"],
            acquisition_difficulty=0.6
        ))
        
        rules.append(GrammaticalRule(
            name="complex_negation",
            description="Using don't, doesn't, didn't for negation",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            categories=[GrammaticalCategory.NEGATION, GrammaticalCategory.AUXILIARY],
            examples=["I don't want", "She doesn't like", "He didn't go"],
            acquisition_difficulty=0.6
        ))
        
        rules.append(GrammaticalRule(
            name="conjunction_usage",
            description="Using conjunctions like and, but, because",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            categories=[GrammaticalCategory.CONJUNCTION],
            examples=["I want milk and cookie", "I'm sad because you left"],
            acquisition_difficulty=0.6
        ))
        
        rules.append(GrammaticalRule(
            name="complex_questions",
            description="Forming more complex questions with auxiliary inversion",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.AUXILIARY],
            examples=["Why is the sky blue?", "Where did you go?"],
            acquisition_difficulty=0.7
        ))
        
        # Advanced rules
        rules.append(GrammaticalRule(
            name="complex_clauses",
            description="Using relative and subordinate clauses",
            min_stage=LanguageDevelopmentStage.ADVANCED,
            categories=[GrammaticalCategory.CONJUNCTION],
            examples=["The ball that I was playing with is red", "I think that she likes me"],
            acquisition_difficulty=0.8
        ))
        
        rules.append(GrammaticalRule(
            name="conditional_sentences",
            description="Using if-then constructions",
            min_stage=LanguageDevelopmentStage.ADVANCED,
            categories=[GrammaticalCategory.CONJUNCTION, GrammaticalCategory.TENSE],
            examples=["If you give me the toy, I will share my cookie", "I would go if I could"],
            acquisition_difficulty=0.8
        ))
        
        rules.append(GrammaticalRule(
            name="passive_voice",
            description="Using passive voice constructions",
            min_stage=LanguageDevelopmentStage.ADVANCED,
            categories=[GrammaticalCategory.VERB, GrammaticalCategory.AUXILIARY],
            examples=["The ball was thrown by me", "The cookie was eaten"],
            acquisition_difficulty=0.9
        ))
        
        return rules
    
    def update_rule_masteries(self, stage: LanguageDevelopmentStage, 
                             grammar_complexity: float) -> None:
        """Update rule masteries based on developmental stage"""
        # Only consider rules appropriate for the current or earlier stages
        applicable_rules = [rule for rule in self.grammar_rules if rule.min_stage <= stage]
    
    for rule in applicable_rules:
        if rule.mastery_level < 0.9:  # If not already mastered
            # Calculate learning potential based on rule difficulty and grammar complexity
            learning_potential = max(0.0, grammar_complexity - rule.acquisition_difficulty)
            
            if learning_potential > 0:
                # Apply learning with random factor
                learning_increment = learning_potential * 0.05 * random.uniform(0.5, 1.5)
                rule.mastery_level = min(0.95, rule.mastery_level + learning_increment)
                
                if rule.mastery_level > 0.5 and not rule.first_used:
                    # Mark as first used when mastery reaches threshold
                    rule.first_used = datetime.now()
                    logger.info(f"Grammar rule '{rule.name}' reached usable mastery level")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for grammatical patterns and extract syntactic information"""
        result = {
            "pos_tags": [],
            "sentence_structure": "unknown",
            "grammatical_features": [],
            "complexity": 0.0
        }
        
        if not text or len(text.strip()) == 0:
            return result
            
        if self.spacy_available and self.nlp:
            # Use spaCy for advanced analysis
            doc = self.nlp(text)
            
            # Extract POS tags
            result["pos_tags"] = [(token.text, token.pos_) for token in doc]
            
            # Determine sentence structure
            if len(doc) == 1:
                result["sentence_structure"] = "single_word"
            elif len(doc) <= 3 and all(token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] for token in doc):
                result["sentence_structure"] = "telegraphic"
            elif any(token.dep_ == "ROOT" and token.pos_ == "VERB" for token in doc):
                # Check for subject-verb structure
                if any(token.dep_ == "nsubj" for token in doc):
                    if any(token.dep_ == "dobj" for token in doc):
                        result["sentence_structure"] = "subject_verb_object"
                    else:
                        result["sentence_structure"] = "subject_verb"
                else:
                    result["sentence_structure"] = "verb_based"
            
            # Extract grammatical features
            features = []
            
            # Check for tense
            has_past = any(token.tag_ in ["VBD", "VBN"] for token in doc)
            has_future = any(token.lower_ in ["will", "going"] for token in doc)
            if has_past:
                features.append("past_tense")
            elif has_future:
                features.append("future_tense")
            else:
                features.append("present_tense")
            
            # Check for negation
            if any(token.dep_ == "neg" for token in doc):
                features.append("negation")
                
            # Check for questions
            if text.strip().endswith("?"):
                features.append("question")
                if any(token.tag_ in ["WDT", "WP", "WP$", "WRB"] for token in doc):
                    features.append("wh_question")
                elif any(token.dep_ == "aux" and token.i == 0 for token in doc):
                    features.append("yes_no_question")
            
            # Check for conjunctions
            if any(token.pos_ == "CCONJ" for token in doc):
                features.append("coordination")
            if any(token.pos_ == "SCONJ" for token in doc):
                features.append("subordination")
            
            # Calculate complexity score based on various factors
            complexity = min(1.0, (
                len(doc) * 0.01 +  # Length factor
                len(set(token.pos_ for token in doc)) * 0.05 +  # POS diversity
                (0.2 if "coordination" in features else 0) +
                (0.3 if "subordination" in features else 0) +
                (0.1 if "question" in features else 0) +
                (0.1 if len([t for t in doc if t.pos_ == "VERB"]) > 1 else 0)  # Multiple verbs
            ))
            
            result["grammatical_features"] = features
            result["complexity"] = complexity
            
            return result
        else:
            # Simplified analysis for when spaCy is not available
            words = text.strip().split()
            
            # Very basic POS tagging
            pos_tags = []
            for word in words:
                # Extremely simplistic POS guess
                if word.lower() in ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]:
                    pos_tags.append((word, "PRON"))
                elif word.lower() in ["a", "an", "the"]:
                    pos_tags.append((word, "DET"))
                elif word.lower() in ["is", "am", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did"]:
                    pos_tags.append((word, "AUX"))
                elif word.lower().endswith("ly"):
                    pos_tags.append((word, "ADV"))
                elif word.lower().endswith(("ed", "ing")):
                    pos_tags.append((word, "VERB"))
                else:
                    # Default to noun for simplicity
                    pos_tags.append((word, "NOUN"))
            
            result["pos_tags"] = pos_tags
            
            # Extremely simplistic sentence structure detection
            if len(words) == 1:
                result["sentence_structure"] = "single_word"
            elif len(words) <= 3:
                result["sentence_structure"] = "telegraphic"
            else:
                result["sentence_structure"] = "multi_word"
            
            # Basic features
            features = []
            if "?" in text:
                features.append("question")
            if "not" in words or "no" in words:
                features.append("negation")
            if "and" in words:
                features.append("coordination")
            if any(w in words for w in ["because", "if", "when", "while"]):
                features.append("subordination")
                
            result["grammatical_features"] = features
            result["complexity"] = min(1.0, len(words) * 0.05)
            
            return result
    
    def apply_grammar_rules(self, text: str, stage: LanguageDevelopmentStage) -> str:
        """Apply appropriate grammar rules to correct or modify text"""
        if not text or stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            return text
            
        # Only apply rules that have sufficient mastery for the current stage
        applicable_rules = [rule for rule in self.grammar_rules 
                          if rule.min_stage <= stage and rule.mastery_level > 0.5]
        
        if not applicable_rules:
            return text
            
        # This is a simplified implementation - in a real system, this would involve
        # complex grammatical transformations based on the rules
        
        if self.spacy_available and self.nlp:
            doc = self.nlp(text)
            words = [token.text for token in doc]
            
            modified = False
            
            # Apply rules based on the grammatical structure
            for rule in applicable_rules:
                # Apply rule with probability based on mastery level
                if random.random() < rule.mastery_level:
                    # Simple example: if we have subject-verb agreement rule
                    if rule.name == "subject_verb_agreement":
                        # Find subject and verb
                        for token in doc:
                            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                                # In a real implementation, we would correct the verb form
                                modified = True
                    
                    # More rule applications would go here...
                    
                    if modified:
                        rule.update_mastery(True)
                    
            return text  # In a real implementation, we would return the modified text
        else:
            # Without spaCy, just return the original text
            return text
    
    def apply_developmental_errors(self, text: str, stage: LanguageDevelopmentStage, 
                                 grammar_complexity: float) -> str:
        """Apply age-appropriate grammatical errors to make speech realistic"""
        if not text or stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            return text
            
        # Get error patterns for this stage
        error_patterns = self.error_patterns.get(stage, {})
        if not error_patterns:
            return text
            
        # The higher the grammar complexity, the fewer errors
        error_reduction = grammar_complexity
        
        words = text.split()
        if len(words) <= 1:
            return text
            
        modified_words = words.copy()
        
        # Apply word omission errors (especially function words)
        if "word_omission" in error_patterns and random.random() < error_patterns["word_omission"] * (1 - error_reduction):
            # Identify function words (simplified)
            function_word_positions = []
            for i, word in enumerate(words):
                if word.lower() in ["a", "an", "the", "is", "am", "are", "to", "in", "on", "at", "of"]:
                    function_word_positions.append(i)
                    
            if function_word_positions and len(words) > 2:
                # Omit a random function word
                omit_position = random.choice(function_word_positions)
                modified_words.pop(omit_position)
        
        # Apply inflection errors (missing -s, -ed, -ing)
        if "inflection_omission" in error_patterns and random.random() < error_patterns["inflection_omission"] * (1 - error_reduction):
            for i, word in enumerate(modified_words):
                if word.endswith(("s", "ed", "ing")) and len(word) > 3:
                    # Strip inflection with some probability
                    if word.endswith("ing"):
                        modified_words[i] = word[:-3]
                    elif word.endswith("ed") and len(word) > 3:
                        modified_words[i] = word[:-2]
                    elif word.endswith("s") and len(word) > 2:
                        modified_words[i] = word[:-1]
                    break  # Only apply to one word
        
        # Apply word order errors in more complex sentences
        if "word_order" in error_patterns and len(modified_words) > 3 and random.random() < error_patterns["word_order"] * (1 - error_reduction):
            # Swap two adjacent words
            swap_pos = random.randint(0, len(modified_words) - 2)
            modified_words[swap_pos], modified_words[swap_pos + 1] = modified_words[swap_pos + 1], modified_words[swap_pos]
        
        # Apply pronoun errors (me/I confusion)
        if "pronoun_error" in error_patterns and random.random() < error_patterns["pronoun_error"] * (1 - error_reduction):
            for i, word in enumerate(modified_words):
                if word.lower() == "i":
                    modified_words[i] = "me"
                    break
                elif word.lower() == "me" and random.random() < 0.5:
                    modified_words[i] = "I"
                    break
        
        return " ".join(modified_words)
    
    def get_applicable_rules(self, stage: LanguageDevelopmentStage) -> List[GrammaticalRule]:
        """Get grammatical rules applicable to the current developmental stage"""
        return [rule for rule in self.grammar_rules if rule.min_stage <= stage]
    
    def get_mastered_rules(self, min_mastery: float = 0.7) -> List[GrammaticalRule]:
        """Get grammatical rules that have been sufficiently mastered"""
        return [rule for rule in self.grammar_rules if rule.mastery_level >= min_mastery]
    
    def get_grammar_statistics(self) -> Dict[str, Any]:
        """Get statistics about grammatical development"""
        total_rules = len(self.grammar_rules)
        mastered_rules = len(self.get_mastered_rules())
        
        # Group rules by stage
        rules_by_stage = {}
        for stage in LanguageDevelopmentStage:
            stage_rules = [rule for rule in self.grammar_rules if rule.min_stage == stage]
            mastered = len([rule for rule in stage_rules if rule.mastery_level >= 0.7])
            rules_by_stage[stage.value] = {
                "total": len(stage_rules),
                "mastered": mastered,
                "percentage": (mastered / len(stage_rules) * 100) if stage_rules else 0
            }
        
        # Group rules by category
        rules_by_category = {}
        for rule in self.grammar_rules:
            for category in rule.categories:
                if category.value not in rules_by_category:
                    rules_by_category[category.value] = {"total": 0, "mastered": 0}
                rules_by_category[category.value]["total"] += 1
                if rule.mastery_level >= 0.7:
                    rules_by_category[category.value]["mastered"] += 1
        
        for category in rules_by_category:
            rules_by_category[category]["percentage"] = (
                rules_by_category[category]["mastered"] / rules_by_category[category]["total"] * 100
            )
        
        return {
            "total_rules": total_rules,
            "mastered_rules": mastered_rules,
            "mastery_percentage": (mastered_rules / total_rules * 100) if total_rules else 0,
            "by_stage": rules_by_stage,
            "by_category": rules_by_category,
            "recently_mastered": [
                rule.name for rule in self.grammar_rules
                if 0.7 <= rule.mastery_level <= 0.8 and rule.first_used
            ]
        }