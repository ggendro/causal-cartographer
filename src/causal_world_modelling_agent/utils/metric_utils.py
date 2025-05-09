
import re
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    # Try to import sentence transformers for semantic similarity
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Constants
POSITIVE_TERMS = ['positive', 'increase', 'increased', 'growing', 'growth', 'upward',
                 'rising', 'rose', 'higher', 'gain', 'gains', 'improved', 'improving',
                 'up', 'stronger', 'strengthen', 'strengthening', 'expanded', 'expanding']

NEGATIVE_TERMS = ['negative', 'decrease', 'decreased', 'declining', 'decline', 'downward',
                 'falling', 'fell', 'lower', 'loss', 'losses', 'worsen', 'worsening',
                 'down', 'weaker', 'weakening', 'contracted', 'contracting']

NEUTRAL_TERMS = ['neutral', 'unchanged', 'stable', 'steady', 'flat', 'maintained',
                'consistent', 'balanced', 'mixed', 'ambiguous']


class LLMOutputEvaluator:
    """
    Evaluates LLM output against ground truth values.
    
    This class provides methods to assess how well an LLM's free-text output 
    aligns with expected ground truth values of various types.
    """
    
    def __init__(self, use_semantic: bool = True, semantic_model: str = 'all-mpnet-base-v2'):
        """
        Initialize the evaluator.
        
        Args:
            use_semantic: Whether to use semantic similarity (requires sentence-transformers)
            semantic_model: The model to use for semantic similarity
        """
        self.use_semantic = use_semantic and SBERT_AVAILABLE
        self.semantic_model = None
        
        if self.use_semantic:
            try:
                self.semantic_model = SentenceTransformer(semantic_model)
            except Exception as e:
                print(f"Failed to load semantic model: {e}")
                self.use_semantic = False
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def evaluate_numerical(self, llm_output: str, ground_truth: str, 
                          tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate LLM output against a numerical ground truth.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The numerical ground truth value
            tolerance: Relative tolerance for numerical comparison
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract numerical values from the output
        numbers = self._extract_numbers(llm_output)
        ground_truth = self._extract_numbers(ground_truth)
        
        if not numbers or not ground_truth or len(ground_truth) == 0 or len(ground_truth) > 1:
            return {
                "found_value": None,
                "exact_match": False,
                "within_tolerance": False,
                "value_error": None,
                "relative_error": None,
                "directional_match": None,
                "confidence": 0.0
            }
        
        ground_truth = ground_truth[0]  # Take the first ground truth value
        
        # Find the closest number to the ground truth
        closest_number = min(numbers, key=lambda x: abs(x - ground_truth))
        absolute_error = abs(closest_number - ground_truth)
        relative_error = absolute_error / max(1e-10, abs(ground_truth))
        within_tolerance = relative_error <= tolerance
        
        # Directional matching (is the number positive/negative as ground truth?)
        same_sign = (closest_number > 0 and ground_truth > 0) or \
                   (closest_number < 0 and ground_truth < 0) or \
                   (closest_number == 0 and ground_truth == 0)
        
        # Calculate confidence based on presence of numerical values and context
        confidence = self._calculate_confidence(llm_output, closest_number, ground_truth)
        
        return {
            "found_value": closest_number,
            "exact_match": closest_number == ground_truth,
            "within_tolerance": within_tolerance,
            "value_error": absolute_error,
            "relative_error": relative_error,
            "directional_match": same_sign,
            "confidence": confidence
        }
    
    def evaluate_categorical(self, llm_output: str, 
                           ground_truth: str, 
                           possible_values: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate LLM output against a categorical ground truth.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The categorical ground truth value
            possible_values: List of possible categorical values (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Clean and normalize text
        normalized_output = self._preprocess_text(llm_output)
        normalized_truth = self._preprocess_text(ground_truth)
        
        # Direct matching
        exact_match = normalized_truth in normalized_output
        
        # Keyword matching for possible values
        detected_categories = []
        if possible_values:
            for value in possible_values:
                norm_value = self._preprocess_text(value)
                if norm_value in normalized_output:
                    detected_categories.append(value)
        
        # Semantic similarity
        semantic_similarity = 0.0
        if self.use_semantic and self.semantic_model:
            # Compare output to ground truth
            similarity = util.cos_sim(
                self.semantic_model.encode(llm_output),
                self.semantic_model.encode(ground_truth)
            ).item()
            semantic_similarity = similarity
        
        # Calculate confidence
        confidence = self._calculate_confidence(llm_output, None, ground_truth)
        if semantic_similarity > 0:
            confidence = max(confidence, semantic_similarity)
        
        return {
            "exact_match": exact_match,
            "detected_categories": detected_categories,
            "correctly_detected": ground_truth in detected_categories,
            "semantic_similarity": semantic_similarity,
            "confidence": confidence
        }
    
    def evaluate_boolean(self, llm_output: str, ground_truth: bool) -> Dict[str, Any]:
        """
        Evaluate LLM output against a boolean ground truth.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The boolean ground truth value
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Look for true/false indicators
        true_pattern = r'\b(true|yes|correct|right|positive|affirm|confirm)\b'
        false_pattern = r'\b(false|no|incorrect|wrong|negative|deny|reject)\b'
        
        true_matches = len(re.findall(true_pattern, llm_output.lower()))
        false_matches = len(re.findall(false_pattern, llm_output.lower()))
        
        # Determine predicted value
        predicted = None
        if true_matches > false_matches:
            predicted = True
        elif false_matches > true_matches:
            predicted = False
        
        # Calculate match and confidence
        match = predicted == ground_truth if predicted is not None else None
        
        # Calculate confidence based on pattern strength
        if predicted is not None:
            # Confidence based on how strongly one pattern predominates
            total_matches = true_matches + false_matches
            if total_matches > 0:
                dominant_matches = true_matches if predicted else false_matches
                confidence = dominant_matches / total_matches
            else:
                confidence = 0.5  # Equal matches, uncertain
        else:
            confidence = 0.0  # No matches, very uncertain
        
        return {
            "predicted": predicted,
            "match": match,
            "true_indicators": true_matches,
            "false_indicators": false_matches,
            "confidence": confidence
        }
    
    def evaluate_directional(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate directional alignment (increase/decrease, positive/negative).
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The directional ground truth value
            
        Returns:
            Dictionary with evaluation metrics
        """
        output_lower = llm_output.lower()
        
        # Count directional terms
        positive_count = sum(1 for term in POSITIVE_TERMS if term in output_lower)
        negative_count = sum(1 for term in NEGATIVE_TERMS if term in output_lower)
        neutral_count = sum(1 for term in NEUTRAL_TERMS if term in output_lower)
        
        # Determine dominant direction in output
        if positive_count > negative_count and positive_count > neutral_count:
            output_direction = "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            output_direction = "negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            output_direction = "neutral"
        else:
            output_direction = "mixed"
        
        # Normalize ground truth - common directional terms
        truth_lower = ground_truth.lower()
        
        if any(term in truth_lower for term in POSITIVE_TERMS):
            truth_direction = "positive"
        elif any(term in truth_lower for term in NEGATIVE_TERMS):
            truth_direction = "negative"
        elif any(term in truth_lower for term in NEUTRAL_TERMS):
            truth_direction = "neutral"
        else:
            # If no clear indicators, use basic keywords
            if any(w in truth_lower for w in ["up", "increase", "higher", "rise"]):
                truth_direction = "positive"
            elif any(w in truth_lower for w in ["down", "decrease", "lower", "drop"]):
                truth_direction = "negative"
            else:
                truth_direction = "unknown"
        
        # Calculate match
        match = output_direction == truth_direction
        
        # Calculate confidence
        total_terms = positive_count + negative_count + neutral_count
        if total_terms > 0:
            if output_direction == "positive":
                confidence = positive_count / total_terms
            elif output_direction == "negative":
                confidence = negative_count / total_terms
            elif output_direction == "neutral":
                confidence = neutral_count / total_terms
            else:  # mixed
                confidence = 0.5
        else:
            confidence = 0.0
        
        return {
            "output_direction": output_direction,
            "truth_direction": truth_direction,
            "match": match,
            "positive_terms": positive_count,
            "negative_terms": negative_count,
            "neutral_terms": neutral_count,
            "confidence": confidence
        }
    
    def evaluate_text(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate general text response.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The ground truth text
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Preprocess texts
        processed_output = self._preprocess_text(llm_output)
        processed_truth = self._preprocess_text(ground_truth)
        
        # Calculate token overlap
        output_tokens = set(word_tokenize(processed_output))
        truth_tokens = set(word_tokenize(processed_truth))
        
        if truth_tokens:
            overlap_ratio = len(output_tokens.intersection(truth_tokens)) / len(truth_tokens)
        else:
            overlap_ratio = 0.0
        
        # Calculate semantic similarity
        semantic_similarity = 0.0
        if self.use_semantic and self.semantic_model:
            similarity = util.cos_sim(
                self.semantic_model.encode(llm_output),
                self.semantic_model.encode(ground_truth)
            ).item()
            semantic_similarity = similarity
        
        # Combined score - average of overlap and semantic similarity
        if self.use_semantic:
            combined_score = (overlap_ratio + semantic_similarity) / 2
        else:
            combined_score = overlap_ratio
        
        return {
            "token_overlap": overlap_ratio,
            "semantic_similarity": semantic_similarity,
            "combined_score": combined_score
        }
    
    def detect_ground_truth_type(self, value: str) -> str:
        """
        Attempts to detect the type of the ground truth value.
        
        Args:
            value: The ground truth value as a string
            
        Returns:
            String indicating the detected type
        """
        # Check if it's a boolean
        if value.lower() in ['true', 'false', 't', 'f', 'yes', 'no']:
            return 'boolean'
            
        # Check if it's a number
        try:
            float(value)
            return 'numerical'
        except ValueError:
            pass
        
        # Check for directional indicators
        value_lower = value.lower()
        if any(term in value_lower for term in POSITIVE_TERMS + NEGATIVE_TERMS):
            return 'directional'
            
        # Check for categorical values - common categorical patterns
        categorical_patterns = [
            r'\b(high|medium|low)\b',
            r'\b(small|medium|large)\b',
            r'\b(positive|negative|neutral)\b',
            r'\b(good|bad|moderate)\b',
            r'\b(increased?|decreased?|unchanged)\b'
        ]
        
        for pattern in categorical_patterns:
            if re.search(pattern, value_lower):
                return 'categorical'
        
        # Default to text
        return 'text'
    
    def evaluate_auto(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Automatically detect ground truth type and evaluate accordingly.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The ground truth value
            
        Returns:
            Dictionary with evaluation metrics
        """
        llm_output = str(llm_output).strip()
        ground_truth = str(ground_truth).strip()

        # Detect type
        detected_type = self.detect_ground_truth_type(ground_truth)
        print(llm_output, ground_truth, detected_type)
        
        # Evaluate based on detected type
        if detected_type == 'numerical':
            result = self.evaluate_numerical(llm_output, ground_truth)
        elif detected_type == 'boolean':
            # Convert string to boolean
            bool_value = ground_truth.lower() in ['true', 't', 'yes', '1']
            result = self.evaluate_boolean(llm_output, bool_value)
        elif detected_type == 'directional':
            result = self.evaluate_directional(llm_output, ground_truth)
        elif detected_type == 'categorical':
            result = self.evaluate_categorical(llm_output, ground_truth)
        else:
            result = self.evaluate_text(llm_output, ground_truth)
        
        # Add detected type to result
        result['detected_type'] = detected_type
        
        return result
    
    def batch_evaluate(self, data: Dict | pd.DataFrame, 
                      llm_output_col: str, 
                      ground_truth_col: str) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Evaluate a batch of LLM outputs against ground truths.
        
        Args:
            data: DataFrame or dictionary with outputs and ground truths
            llm_output_col: Column name for LLM outputs
            ground_truth_col: Column name for ground truths
            
        Returns:
            DataFrame with evaluation results added or list of result dictionaries
        """
        results = []
        
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
            
        for _, row in df.iterrows():
            llm_output = row[llm_output_col]
            ground_truth = row[ground_truth_col]
            
            # Evaluate
            eval_result = self.evaluate_auto(llm_output, ground_truth)
            
            # Add to results
            results.append(eval_result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Concatenate with original DataFrame
        if isinstance(data, pd.DataFrame):
            return pd.concat([df, results_df], axis=1)
        else:
            return results
    
    def summarize_results(self, results: List[Dict[str, Any]] | pd.DataFrame) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        
        Args:
            results: List of result dictionaries or DataFrame with results
            
        Returns:
            Dictionary with summary metrics
        """
        
        # Convert list to DataFrame if needed
        if not isinstance(results, pd.DataFrame):
            results_df = pd.DataFrame(results)
        else:
            results_df = results
            
        summary = {}
        
        # Count by type
        if 'detected_type' in results_df:
            type_counts = results_df['detected_type'].value_counts().to_dict()
            summary['type_counts'] = type_counts
        
        # Average confidence
        if 'confidence' in results_df:
            summary['avg_confidence'] = results_df['confidence'].mean()
        
        # Matches by type
        if 'match' in results_df and 'detected_type' in results_df:
            match_by_type = results_df.groupby('detected_type')['match'].mean().to_dict()
            summary['match_by_type'] = match_by_type
        
        # Overall match rate
        if 'match' in results_df:
            summary['overall_match_rate'] = results_df['match'].mean()
        
        # Semantic similarity stats
        if 'semantic_similarity' in results_df:
            summary['avg_semantic_similarity'] = results_df['semantic_similarity'].mean()
        
        return summary
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Pattern for numbers (including decimals, negatives, percentages)
        pattern = r'[-+]?\d*\.?\d+(?:%|percent)?'
        matches = re.findall(pattern, text)
        
        # Convert to float (removing % if present)
        numbers = []
        for match in matches:
            if '%' in match or 'percent' in match:
                # Convert percentage to decimal
                match = match.replace('%', '').replace('percent', '')
                numbers.append(float(match) / 100)
            else:
                numbers.append(float(match))
                
        return numbers
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return " ".join(tokens)
    
    def _calculate_confidence(self, text: str, 
                            predicted_value: Optional[float] = None, 
                            ground_truth: Optional[Any] = None) -> float:
        """Calculate confidence in the evaluation."""
        # Basic confidence based on text length (longer answers may be more confident)
        base_confidence = min(len(text) / 500, 0.7)  # Cap at 0.7
        
        # If we have a predicted numerical value
        if predicted_value is not None and isinstance(ground_truth, (int, float)):
            # Confidence decreases as the relative error increases
            relative_error = abs(predicted_value - ground_truth) / max(1e-10, abs(ground_truth))
            error_confidence = max(0, 1 - min(relative_error, 1))
            
            # Combine base and error confidence
            return 0.3 * base_confidence + 0.7 * error_confidence
        
        return base_confidence





def visualize_evaluation_results(results_df: pd.DataFrame, 
                               output_path: Optional[str] = None,
                               show_plot: bool = True) -> None:
    """
    Create visualizations for evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results
        output_path: Path to save the visualization
        show_plot: Whether to display the plot
    """
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    n_plots = 3
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 15))
    
    # 1. Confidence distribution by type
    if 'confidence' in results_df.columns and 'detected_type' in results_df.columns:
        sns.boxplot(x='detected_type', y='confidence', data=results_df, ax=axs[0])
        axs[0].set_title('Confidence Distribution by Ground Truth Type')
        axs[0].set_xlabel('Ground Truth Type')
        axs[0].set_ylabel('Confidence Score')
    
    # 2. Match rate by type
    if 'match' in results_df.columns and 'detected_type' in results_df.columns:
        match_by_type = results_df.groupby('detected_type')['match'].mean().reset_index()
        sns.barplot(x='detected_type', y='match', data=match_by_type, ax=axs[1])
        axs[1].set_title('Match Rate by Ground Truth Type')
        axs[1].set_xlabel('Ground Truth Type')
        axs[1].set_ylabel('Match Rate')
        axs[1].set_ylim(0, 1)
    
    # 3. Semantic similarity distribution
    if 'semantic_similarity' in results_df.columns:
        sns.histplot(results_df['semantic_similarity'], kde=True, ax=axs[2])
        axs[2].set_title('Semantic Similarity Distribution')
        axs[2].set_xlabel('Semantic Similarity Score')
        axs[2].set_ylabel('Count')
        axs[2].set_xlim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
    
    if show_plot:
        plt.show()
    else:
        plt.close()




__all__ = [
    'LLMOutputEvaluator',
    'visualize_evaluation_results'
]