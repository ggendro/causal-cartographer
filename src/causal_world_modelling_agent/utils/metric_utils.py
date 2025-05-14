import re
from typing import Dict, List, Union, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast
import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix

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
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Constants
POSITIVE_TERMS = ['positive', 'increase', 'increased', 'growing', 'growth', 'upward',
                 'rising', 'rose', 'higher', 'gain', 'gains', 'improved', 'improving',
                 'up', 'stronger', 'strengthen', 'strengthening', 'expanded', 'expanding',
                 'appreciation', 'appreciating', 'elevated', 'bullish', 'rally', 'rise']

NEGATIVE_TERMS = ['negative', 'decrease', 'decreased', 'declining', 'decline', 'downward',
                 'falling', 'fell', 'lower', 'loss', 'losses', 'worsen', 'worsening',
                 'down', 'weaker', 'weakening', 'contracted', 'contracting', 
                 'depreciation', 'depreciating', 'depreciated', 'bearish', 'slump', 'drop']

NEUTRAL_TERMS = ['neutral', 'unchanged', 'stable', 'steady', 'flat', 'maintained',
                'consistent', 'balanced', 'mixed', 'ambiguous', 'moderate', 'moderated']

FINANCIAL_ASSETS = ['brent', 'wti', 'crude', 'oil', 'gold', 'dollar', 'rupee', 'usd', 'eur', 
                   'euro', 'stock', 'stocks', 'equity', 'equities', 'index', 'indices', 
                   'shares', 'bond', 'bonds', 'treasury', 'yield', 'yields', 'interest',
                   'pound', 'pounds', 'yen', 'won', 'peso', 'pesos', 'franc', 'francs',
                   '€', '$', '£', '¥', '₩', '₹', '₽', '₺', '₦', '₵', '₭', '₮']



# decorator ading prefix to keys in dict
def add_prefix(prefix: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return {f"{prefix}_{k}": v for k, v in result.items()}
        return wrapper
    return decorator


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
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    
    @add_prefix('numerical')
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
        
        if not numbers or not ground_truth or len(ground_truth) == 0:
            return {
                "found_value": None,
                "exact_match": False,
                "within_tolerance": False,
                "value_error": None,
                "relative_error": None,
                "directional_match": None,
                "confidence": 0.0
            }
        # Take the first values
        ground_truth = ground_truth[0]
        predicted_value = numbers[0]

        # Find the closest number to the ground truth
        absolute_error = abs(predicted_value - ground_truth)
        relative_error = absolute_error / max(1e-10, abs(ground_truth))
        within_tolerance = relative_error <= tolerance
        
        # Directional matching (is the number positive/negative as ground truth?)
        same_sign = predicted_value * ground_truth >= 0
        
        return {
            "found_value": predicted_value,
            "exact_match": predicted_value == ground_truth,
            "within_tolerance": within_tolerance,
            "value_error": absolute_error,
            "relative_error": relative_error,
            "directional_match": same_sign
        }
    
    @add_prefix('boolean')
    def evaluate_boolean(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
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

        gt_true_matches = len(re.findall(true_pattern, ground_truth.lower()))
        gt_false_matches = len(re.findall(false_pattern, ground_truth.lower()))

        if not (bool(gt_true_matches) ^ bool(gt_false_matches)):
            # If both or neither are present, we can't determine the ground truth
            return {
                "predicted": None,
                "match": None,
                "true_indicators": gt_true_matches,
                "false_indicators": gt_false_matches,
                "confidence": 0.0
            }
        else:
            # Determine the ground truth value based on the stronger indicator
            ground_truth = gt_true_matches > gt_false_matches

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
    
    @add_prefix('directional')
    def evaluate_directional(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate directional alignment (increase/decrease, positive/negative).
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The directional ground truth value
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Empty result dictionary for numerical content
        empty_result = {
            "output_direction": None, "truth_direction": None, "match": None,
            "positive_terms": 0, "negative_terms": 0, "neutral_terms": 0,
            "confidence": 0.0, "sentiment_compound": None, "sentiment_positive": None,
            "sentiment_negative": None, "sentiment_neutral": None,
            "keyword_direction": None, "sentiment_direction": None
        }
        
        # Check if either text is primarily numerical without directional indicators
        output_lower = llm_output.lower()
        truth_lower = ground_truth.lower()
        
        # Simple function to determine if text is primarily numerical
        def is_numerical_only(text):
            numbers = self._extract_numbers(text)
            directional_terms = sum(1 for term in POSITIVE_TERMS + NEGATIVE_TERMS + NEUTRAL_TERMS if term in text)
            return len(numbers) > 0 and directional_terms == 0
        
        # Skip directional evaluation if either input is purely numerical
        if is_numerical_only(output_lower) or is_numerical_only(truth_lower):
            return empty_result
        
        # Filter out numerical content for analysis
        filtered_output = self._filter_numerical_content(output_lower)
        filtered_truth = self._filter_numerical_content(truth_lower)
        
        # Use VADER sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(filtered_output)
        
        # Count directional terms
        positive_count = sum(1 for term in POSITIVE_TERMS if term in output_lower)
        negative_count = sum(1 for term in NEGATIVE_TERMS if term in output_lower)
        neutral_count = sum(1 for term in NEUTRAL_TERMS if term in output_lower)
        
        # Determine sentiment direction
        sentiment_direction = "neutral"
        if sentiment_scores['compound'] >= 0.05:
            sentiment_direction = "positive"
        elif sentiment_scores['compound'] <= -0.05:
            sentiment_direction = "negative"
            
        # Determine keyword direction
        if positive_count > negative_count and positive_count > neutral_count:
            keyword_direction = "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            keyword_direction = "negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            keyword_direction = "neutral"
        else:
            keyword_direction = "mixed"
            
        # Determine final output direction with sentiment as tiebreaker
        output_direction = sentiment_direction if (keyword_direction in ["mixed", "neutral"]) else (
            keyword_direction if keyword_direction == sentiment_direction or abs(sentiment_scores['compound']) <= 0.5
            else sentiment_direction
        )
        
        # Determine ground truth direction using same strategy as output
        truth_sentiment = self.sentiment_analyzer.polarity_scores(filtered_truth)
        
        # Count directional terms in ground truth
        truth_positive = any(term in truth_lower for term in POSITIVE_TERMS)
        truth_negative = any(term in truth_lower for term in NEGATIVE_TERMS)
        truth_neutral = any(term in truth_lower for term in NEUTRAL_TERMS)
        
        # Determine truth direction using keywords, falling back to sentiment
        if truth_positive and not (truth_negative or truth_neutral):
            truth_direction = "positive"
        elif truth_negative and not (truth_positive or truth_neutral):
            truth_direction = "negative"
        elif truth_neutral and not (truth_positive or truth_negative):
            truth_direction = "neutral"
        else:
            # Use sentiment for mixed keyword signals or no keywords
            if truth_sentiment['compound'] >= 0.05:
                truth_direction = "positive"
            elif truth_sentiment['compound'] <= -0.05:
                truth_direction = "negative"
            else:
                truth_direction = "neutral"
        
        # Calculate match
        match = output_direction == truth_direction
        
        # Calculate confidence (simplified)
        total_terms = positive_count + negative_count + neutral_count
        # Choose relevant count based on output direction
        relevant_count = (positive_count if output_direction == "positive" 
                         else negative_count if output_direction == "negative"
                         else neutral_count if output_direction == "neutral"
                         else total_terms // 2)  # For mixed, use half
        
        # Calculate keyword and sentiment confidence
        keyword_confidence = relevant_count / total_terms if total_terms > 0 else 0.0
        
        # Calculate sentiment confidence (normalized)
        sentiment_confidence = abs(sentiment_scores['compound'])
        
        # Combined confidence (weighted average)
        confidence = (0.6 * keyword_confidence) + (0.4 * sentiment_confidence) if total_terms > 0 else sentiment_confidence
        
        # Return concise result dictionary
        return {
            "output_direction": output_direction,
            "truth_direction": truth_direction,
            "match": match,
            "positive_terms": positive_count,
            "negative_terms": negative_count,
            "neutral_terms": neutral_count,
            "confidence": confidence,
            "sentiment_compound": sentiment_scores['compound'],
            "sentiment_positive": sentiment_scores['pos'],
            "sentiment_negative": sentiment_scores['neg'],
            "sentiment_neutral": sentiment_scores['neu'],
            "keyword_direction": keyword_direction,
            "sentiment_direction": sentiment_direction
        }
    
    @add_prefix('text')
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
        
        # Tokenize for BLEU calculation
        output_tokens = word_tokenize(processed_output)
        truth_tokens = word_tokenize(processed_truth)
        
        # Calculate BLEU score with smoothing
        bleu_score = 0.0
        if truth_tokens:
            try:
                # Use smoothing to avoid zero scores due to n-gram precision issues
                bleu_score = sentence_bleu([truth_tokens], output_tokens, smoothing_function=SmoothingFunction().method1)
            except Exception as e:
                print(f"BLEU score calculation error: {e}")
                bleu_score = 0.0
        
        # Calculate semantic similarity
        semantic_similarity = 0.0
        if self.use_semantic and self.semantic_model:
            similarity = util.cos_sim(
                self.semantic_model.encode(llm_output),
                self.semantic_model.encode(ground_truth)
            ).item()
            semantic_similarity = similarity
        
        # Combined score - average of BLEU and semantic similarity
        if self.use_semantic:
            combined_score = (bleu_score + semantic_similarity) / 2
        else:
            combined_score = bleu_score
        
        return {
            "bleu_score": bleu_score,
            "semantic_similarity": semantic_similarity,
            "combined_score": combined_score
        }
    
    @add_prefix('financial')
    def evaluate_financial_price(self, llm_output: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate LLM output against a financial price ground truth.
        Handles cases where the model output might be qualitative while ground truth is numeric.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The financial price ground truth value
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract asset types and directional language
        output_lower = llm_output.lower()
        truth_lower = ground_truth.lower()
        
        # Extract mentioned assets
        output_assets = [asset for asset in FINANCIAL_ASSETS if asset in output_lower]
        truth_assets = [asset for asset in FINANCIAL_ASSETS if asset in truth_lower]
        
        # Check asset match
        assets_match = len(set(output_assets).intersection(set(truth_assets))) > 0
            
        if assets_match:
            return {
                "assets_match": assets_match,
                "output_assets": output_assets,
                "truth_assets": truth_assets
            }
        else:
            # If no asset match, we can't evaluate the financial aspect
            return {
                "assets_match": None,
                "output_assets": None,
                "truth_assets": None
            }
    
    @add_prefix('counterfactual')
    def evaluate_counterfactual(self, llm_output: str, ground_truth: str, 
                                target_observation: str) -> Dict[str, Any]:
        """
        Evaluate counterfactual queries, comparing how well the LLM output captures
        the change from target_observation to ground_truth.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The ground truth value after intervention
            target_observation: The baseline value before intervention
            
        Returns:
            Dictionary with evaluation metrics
        """
            
        # For numerical values, calculate the true directional change
        ground_truth_nums = self._extract_numbers(ground_truth)
        target_nums = self._extract_numbers(target_observation)
        
        # If we have numbers in both ground truth and target, determine the true direction
        if ground_truth_nums and target_nums:
            gt_num = ground_truth_nums[0]
            target_num = target_nums[0]
            
            # Calculate the direction and magnitude of change
            true_change = gt_num - target_num
            true_pct_change = true_change / abs(target_num) if target_num != 0 else float('inf')
            
            if true_change > 0:
                true_direction = "positive"
            elif true_change < 0:
                true_direction = "negative"
            else:
                true_direction = "neutral"
                
            # Get directional evaluation of the output
            directional_result = self.evaluate_directional(llm_output, true_direction)
            output_direction = directional_result["directional_output_direction"]
            direction_match = output_direction == true_direction if output_direction is not None else False
            
            # Extract any numbers from the output and compute trend
            output_nums = self._extract_numbers(llm_output)
            if output_nums:
                num_value = output_nums[0]

                pred_change = num_value - target_num
                pred_pct_change = pred_change / abs(target_num) if target_num != 0 else float('inf')
                
                change_sign_match = (pred_change > 0) == (true_change > 0)
                change_magnitude_error = abs(pred_change) / abs(true_change) if true_change != 0 else float('inf')
                change_pct_error = abs(pred_pct_change - true_pct_change) / abs(true_pct_change) if true_pct_change != 0 else float('inf')
            
            # Return combined metrics
            return {
                "true_change": true_change,
                "true_pct_change": true_pct_change,
                "true_direction": true_direction,
                "output_direction": output_direction,
                "direction_match": direction_match,
                "pred_change": pred_change if output_nums else None,
                "pred_pct_change": pred_pct_change if output_nums else None,
                "change_sign_match": change_sign_match if output_nums else None,
                "change_magnitude_error": change_magnitude_error if output_nums else None,
                "change_pct_error": change_pct_error if output_nums else None
            }
        
        else:
            # If we don't have numbers, we can't evaluate the counterfactual
            return {
                "true_change": None,
                "true_pct_change": None,
                "true_direction": None,
                "output_direction": None,
                "direction_match": None,
                "pred_change": None,
                "pred_pct_change": None,
                "change_sign_match": None,
                "change_magnitude_error": None,
                "change_pct_error": None
            }
    
    
    def evaluate_auto(self, llm_output: str, ground_truth: str, target_observation: str = None) -> Dict[str, Any]:
        """
        Run all evaluation types and combine results appropriately.
        
        Args:
            llm_output: The free-text output from the LLM
            ground_truth: The ground truth value
            target_observation: Optional baseline value for comparison
            
        Returns:
            Dictionary with combined evaluation metrics
        """
        llm_output = str(llm_output).strip()
        ground_truth = str(ground_truth).strip()
        
        # Run all evaluations regardless of detected type
        numerical_result = self.evaluate_numerical(llm_output, ground_truth)
        directional_result = self.evaluate_directional(llm_output, ground_truth)
        text_result = self.evaluate_text(llm_output, ground_truth)
        boolean_result = self.evaluate_boolean(llm_output, ground_truth)
        financial_result = self.evaluate_financial_price(llm_output, ground_truth)
        
        # Add results from different methods
        result = {**numerical_result, **directional_result, **text_result, **boolean_result}
        result["is_numerical"] = numerical_result["numerical_found_value"] is not None
        result["is_boolean"] = boolean_result["boolean_predicted"] is not None
        result["is_directional"] = directional_result["directional_output_direction"] is not None
        result["is_financial"] = financial_result["financial_assets_match"]
        result["is_text"] = True
        
        # If target observation is provided, evaluate as counterfactual
        if target_observation:
            cf_result = self.evaluate_counterfactual(llm_output, ground_truth, target_observation)
            result.update(cf_result)
        
        return result
    
    def batch_evaluate(self, data: Dict | pd.DataFrame, 
                      llm_output_col: str, 
                      ground_truth_col: str,
                      observations_col: str = None,
                      query_type_col: str = None,
                      target_col: str = None) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """
        Evaluate a batch of LLM outputs against ground truths.
        
        Args:
            data: DataFrame or dictionary with outputs and ground truths
            llm_output_col: Column name for LLM outputs
            ground_truth_col: Column name for ground truth values
            observations_col: Column name for target observation values (optional)
            query_type_col: Column name for query type (optional)
            target_col: Column name for target values (optional)
            
        Returns:
            DataFrame with evaluation results added or list of result dictionaries
        """
        results = []
        
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
            
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            if llm_output_col not in row or pd.isna(row[llm_output_col]):
                # Skip rows with missing output
                results.append({
                    "no_output_found": True,
                })
                continue
                
            llm_output = str(row[llm_output_col])
            ground_truth = str(row[ground_truth_col])
            
            # Get target observation if available
            target_observation = None
            if observations_col and target_col and observations_col in row and not pd.isna(row[observations_col]):
                target_name = row[target_col]
                if isinstance(row[observations_col], str):
                    observations = ast.literal_eval(row[observations_col])
                else:
                    observations = row[observations_col]
                for variable in observations:
                    if variable.get("name") == target_name:
                        target_observation = variable.get("current_value")
                        break
            
            # Run comprehensive evaluation
            eval_result = self.evaluate_auto(llm_output, ground_truth, target_observation)
            
            # Add query type if available
            if query_type_col and query_type_col in row:
                eval_result["query_type"] = row[query_type_col]
            
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
        
        # Count by evaluation type
        evaluation_types = ['numerical', 'directional', 'boolean', 'text', 'financial', 'counterfactual']
        type_counts = {}
        
        for eval_type in evaluation_types:
            is_type_col = f"is_{eval_type}"
            if is_type_col in results_df.columns:
                type_counts[eval_type] = int(results_df[is_type_col].sum())
        
        summary['type_counts'] = type_counts
        
        # Overall match rates by type
        match_rates = {}
        
        # Numerical: within tolerance
        if 'is_numerical' in results_df and 'numerical_within_tolerance' in results_df:
            numerical_results = results_df[results_df['is_numerical'] == True]
            if len(numerical_results) > 0:
                match_rates['numerical'] = {
                    'within_tolerance': float(numerical_results['numerical_within_tolerance'].mean()),
                    'exact_match': float(numerical_results['numerical_exact_match'].mean()) if 'numerical_exact_match' in numerical_results else None,
                    'avg_relative_error': float(numerical_results['numerical_relative_error'].mean()) if 'numerical_relative_error' in numerical_results else None
                }
        
        # Directional
        if 'is_directional' in results_df and 'directional_match' in results_df:
            directional_results = results_df[results_df['is_directional'] == True]
            if len(directional_results) > 0:
                match_rates['directional'] = {
                    'direction_match': float(directional_results['directional_match'].mean()),
                    'confidence': float(directional_results['directional_confidence'].mean()) if 'directional_confidence' in directional_results else None
                }
        
        # Boolean
        if 'is_boolean' in results_df and 'boolean_match' in results_df:
            boolean_results = results_df[results_df['is_boolean'] == True]
            if len(boolean_results) > 0:
                match_rates['boolean'] = {
                    'match': float(boolean_results['boolean_match'].mean()),
                    'confidence': float(boolean_results['boolean_confidence'].mean()) if 'boolean_confidence' in boolean_results else None
                }
        
        # Text
        if 'text_combined_score' in results_df:
            match_rates['text'] = {
                'combined_score': float(results_df['text_combined_score'].mean()),
                'bleu_score': float(results_df['text_bleu_score'].mean()) if 'text_bleu_score' in results_df else None,
                'semantic_similarity': float(results_df['text_semantic_similarity'].mean()) if 'text_semantic_similarity' in results_df else None
            }
        
        # Financial
        if 'is_financial' in results_df and 'financial_assets_match' in results_df:
            financial_results = results_df[results_df['is_financial'] == True]
            if len(financial_results) > 0:
                match_rates['financial'] = {
                    'assets_match': float(financial_results['financial_assets_match'].mean())
                }
        
        # Counterfactual
        if 'is_counterfactual' in results_df:
            cf_results = results_df[results_df['is_counterfactual'] == True]
            if len(cf_results) > 0:
                match_rates['counterfactual'] = {
                    'direction_match': float(cf_results['counterfactual_direction_match'].mean()) if 'counterfactual_direction_match' in cf_results else None,
                    'change_sign_match': float(cf_results.dropna(subset=['counterfactual_change_sign_match'])['counterfactual_change_sign_match'].mean()) if 'counterfactual_change_sign_match' in cf_results else None
                }
        
        summary['match_rates'] = match_rates
        
        # Overall statistics
        summary['total_evaluations'] = len(results_df)
        
        # Calculate overall average performance
        # Weight each evaluation type equally
        performance_metrics = []
        
        if 'numerical' in match_rates and match_rates['numerical'].get('within_tolerance') is not None:
            performance_metrics.append(match_rates['numerical']['within_tolerance'])
            
        if 'directional' in match_rates and match_rates['directional'].get('direction_match') is not None:
            performance_metrics.append(match_rates['directional']['direction_match'])
            
        if 'boolean' in match_rates and match_rates['boolean'].get('match') is not None:
            performance_metrics.append(match_rates['boolean']['match'])
            
        if 'text' in match_rates and match_rates['text'].get('combined_score') is not None:
            performance_metrics.append(match_rates['text']['combined_score'])
            
        if 'counterfactual' in match_rates and match_rates['counterfactual'].get('direction_match') is not None:
            performance_metrics.append(match_rates['counterfactual']['direction_match'])
        
        if performance_metrics:
            summary['overall_performance'] = float(sum(performance_metrics) / len(performance_metrics))
        else:
            summary['overall_performance'] = None
        
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
        
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and token.isalnum()]
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return " ".join(tokens)
        
    def _filter_numerical_content(self, text: str) -> str:
        """
        Filter out numerical content from text to improve directional analysis.
        
        Args:
            text: The text to filter
            
        Returns:
            Text with numerical content removed or replaced
        """
        # Replace numbers with placeholders to avoid them affecting sentiment/direction
        # Match numbers with various formats (including percentages and currency symbols)
        currency_pattern = r'[$€£¥]?\s*[-+]?\d*\.?\d+(?:[kmbt])?(?:%|\s*percent)?'
        
        # Replace currency and numbers with placeholders
        filtered_text = re.sub(currency_pattern, " VALUE ", text)
        
        # Remove multiple spaces
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
        
        return filtered_text
    
    
    def plot_evaluation_types(self, results_df: pd.DataFrame, output_dir: str, show_plots: bool = False):
        """
        Plot distribution of evaluation types.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save the plot
            show_plots: Whether to display plots in addition to saving them
        """
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Identify which evaluation types were applied
        evaluation_types = ['numerical', 'directional', 'text', 'boolean', 'financial', 'counterfactual']
        type_counts = {}
        
        for eval_type in evaluation_types:
            is_type_col = f"is_{eval_type}"
            if is_type_col in results_df.columns:
                type_counts[eval_type] = results_df[is_type_col].sum()
        
        # Filter out types with zero count
        type_counts = {k: v for k, v in type_counts.items() if v > 0}
        
        if not type_counts:
            return None
            
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(type_counts.keys(), type_counts.values(), color=sns.color_palette('viridis', len(type_counts)))
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Evaluation Types Applied', fontsize=15)
        plt.xlabel('Evaluation Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(vis_dir, 'evaluation_type_distribution.png')
        plt.savefig(output_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def plot_match_rates(self, results_df: pd.DataFrame, output_dir: str, show_plots: bool = False):
        """
        Plot match rates for each evaluation type.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save the plot
            show_plots: Whether to display plots in addition to saving them
        """
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Calculate match rates for each evaluation type
        evaluation_types = ['numerical', 'directional', 'boolean', 'text', 'counterfactual']
        match_rates = {}
        
        for eval_type in evaluation_types:
            # For numerical, check within_tolerance
            if eval_type == 'numerical' and f"{eval_type}_within_tolerance" in results_df.columns:
                is_type = results_df[f"is_{eval_type}"] == True
                if is_type.any():
                    match_rates[eval_type] = results_df.loc[is_type, f"{eval_type}_within_tolerance"].mean() * 100
            # For text, use combined_score with threshold
            elif eval_type == 'text' and f"{eval_type}_combined_score" in results_df.columns:
                match_rates[eval_type] = results_df[f"{eval_type}_combined_score"].mean() * 100
            # For others, use direct match column
            elif f"{eval_type}_match" in results_df.columns:
                is_type = results_df[f"is_{eval_type}"] == True
                if is_type.any():
                    match_rates[eval_type] = results_df.loc[is_type, f"{eval_type}_match"].mean() * 100
        
        # Remove types with no data
        match_rates = {k: v for k, v in match_rates.items() if not pd.isna(v)}
        
        if not match_rates:
            return None
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(match_rates.keys(), match_rates.values(), color=sns.color_palette('viridis', len(match_rates)))
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Match Rates by Evaluation Type', fontsize=15)
        plt.xlabel('Evaluation Type', fontsize=12)
        plt.ylabel('Match Rate (%)', fontsize=12)
        plt.ylim(0, 105)  # Give space for labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(vis_dir, 'match_rates_by_type.png')
        plt.savefig(output_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        return output_path
    
    def plot_performance_overview(self, results_df: pd.DataFrame, output_dir: str, show_plots: bool = False):
        """
        Create a summary radar chart showing performance across different metrics.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save plots
            show_plots: Whether to display plots in addition to saving them
        """        
        # Create visualizations directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Define key metrics to track
        metrics = {
            'Numerical': results_df[results_df["is_numerical"] == True]['numerical_within_tolerance'].mean() * 100 if 'numerical_within_tolerance' in results_df else None,
            'Directional': results_df[results_df["is_directional"] == True]['directional_match'].mean() * 100 if 'directional_match' in results_df else None,
            'Boolean': results_df[results_df["is_boolean"] == True]['boolean_match'].mean() * 100 if 'boolean_match' in results_df else None,
            'Text': results_df['text_combined_score'].mean() * 100 if 'text_combined_score' in results_df else None,
            'Counterfactual': results_df[results_df["is_counterfactual"] == True]['counterfactual_direction_match'].mean() * 100 if 'counterfactual_direction_match' in results_df else None,
        }
        
        # Filter out None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        if len(metrics) > 0:
            plt.figure(figsize=(8, 8))
            
            # Create radar chart
            categories = list(metrics.keys())
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Add values
            values = list(metrics.values())
            values += values[:1]  # Close the loop
            
            # Draw the plot
            ax = plt.subplot(111, polar=True)
            
            # Draw axis lines for each angle and label
            plt.xticks(angles[:-1], categories, size=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"], size=10, color="grey")
            plt.ylim(0, 100)
            
            # Plot data
            ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
            
            # Fill area
            ax.fill(angles, values, '#3498db', alpha=0.4)
            
            plt.title('Overall Performance Across Metrics', size=15, y=1.1)
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(vis_dir, 'performance_radar.png')
            plt.savefig(output_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            return output_path
        
        return None
    
    
    def _create_subplot_figure(self, n_rows, n_cols, figsize=(12, 10)):
        """Helper method to create a figure with subplots.
        
        Args:
            n_rows: Number of rows in the subplot grid
            n_cols: Number of columns in the subplot grid
            figsize: Figure size as a tuple (width, height)
            
        Returns:
            fig: The matplotlib figure object
            axes: A 2D numpy array of subplot axes
        """
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # If there's only one row or column, wrap axes in a 2D array for consistency
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])
            
        return fig, axes
            
    def plot_numerical_metrics(self, results_df, output_dir, show_plots=False):
        """Plot numerical metrics: value_error and relative_error in merged plots."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'numerical')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Filter for numerical evaluations only
        numerical_results = results_df[results_df["is_numerical"] == True].copy()
        if len(numerical_results) == 0:
            return outputs
        
        # Create a figure with 2 subplots (2 rows, 1 column)
        # One for absolute error, one for relative error
        if 'numerical_value_error' in numerical_results and 'numerical_relative_error' in numerical_results:
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot 1: Value Error
            # Clip extreme values for better visualization
            plot_data = numerical_results.copy()
            plot_data['clipped_value_error'] = plot_data['numerical_value_error'].clip(0, plot_data['numerical_value_error'].quantile(0.95))
            
            sns.violinplot(data=plot_data, y='clipped_value_error', ax=axes[0])
            axes[0].set_title('Distribution of Value Error', fontsize=15)
            axes[0].set_ylabel('Absolute Error', fontsize=12)
            
            # Plot 2: Relative Error
            plot_data['clipped_relative_error'] = plot_data['numerical_relative_error'].clip(0, 2)  # Clip at 200% error
            
            sns.violinplot(data=plot_data, y='clipped_relative_error', ax=axes[1], cut=0)
            axes[1].set_title('Distribution of Relative Error', fontsize=15)
            axes[1].set_ylabel('Relative Error (clipped at 2.0)', fontsize=12)
            
            plt.tight_layout()
            
            # Save combined plot
            output_path = os.path.join(vis_dir, 'error_distributions.png')
            plt.savefig(output_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            outputs['error_distributions'] = output_path
        
        # If we have query_type, create a figure with subplots by type
        if 'query_type' in numerical_results.columns:
            # Filter for just the types we're interested in
            query_types = ['observation', 'counterfactual_match']
            filtered_results = numerical_results[numerical_results['query_type'].isin(query_types)]
            
            if len(filtered_results) > 0 and 'numerical_value_error' in filtered_results and 'numerical_relative_error' in filtered_results:
                fig, axes = plt.subplots(2, 1, figsize=(12, 14))
                
                # Plot 1: Value Error by Query Type
                plot_data = filtered_results.copy()
                plot_data['clipped_value_error'] = plot_data['numerical_value_error'].clip(0, plot_data['numerical_value_error'].quantile(0.95))
                
                sns.violinplot(data=plot_data, x='query_type', y='clipped_value_error', ax=axes[0])
                axes[0].set_title('Value Error by Query Type', fontsize=15)
                axes[0].set_xlabel('Query Type', fontsize=12)
                axes[0].set_ylabel('Absolute Error', fontsize=12)
                
                # Plot 2: Relative Error by Query Type
                plot_data['clipped_relative_error'] = plot_data['numerical_relative_error'].clip(0, 2)
                
                sns.violinplot(data=plot_data, x='query_type', y='clipped_relative_error', ax=axes[1], cut=0)
                axes[1].set_title('Relative Error by Query Type', fontsize=15)
                axes[1].set_xlabel('Query Type', fontsize=12)
                axes[1].set_ylabel('Relative Error (clipped at 2.0)', fontsize=12)
                
                plt.tight_layout()
                
                # Save combined plot
                output_path = os.path.join(vis_dir, 'error_distributions_by_type.png')
                plt.savefig(output_path, dpi=300)
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                    
                outputs['error_distributions_by_type'] = output_path
        
        return outputs
    
    def plot_boolean_metrics(self, results_df, output_dir, show_plots=False):
        """Plot boolean metrics: match rates, confusion matrix and confidence together."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'boolean')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Filter for boolean evaluations only
        boolean_results = results_df[results_df["is_boolean"] == True].copy()
        if len(boolean_results) == 0:
            return outputs
        
        # Create a figure with subplots: 2 rows, 2 columns
        # Top row: Match rate and confusion matrix
        # Bottom row: Confidence and match rate by type
        fig, axes = self._create_subplot_figure(2, 2, figsize=(14, 12))
        
        # Plot 1: Overall match rate (top left)
        if 'boolean_match' in boolean_results:
            match_rate = boolean_results['boolean_match'].mean()
            match_std = boolean_results['boolean_match'].std()
            
            bar = axes[0, 0].bar(['Overall'], [match_rate], yerr=[match_std], 
                                capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[0, 0].text(0, match_rate + 0.02, f'{match_rate:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[0, 0].set_title('Boolean Match Rate', fontsize=15)
            axes[0, 0].set_ylabel('Match Rate', fontsize=12)
            axes[0, 0].set_ylim(0, min(1.0, match_rate + 0.15))  # Leave room for error bar and label
            
            # Plot 2: Confusion matrix (top right)
            if 'boolean_predicted' in boolean_results:
                # Create true values from match and predicted
                boolean_results['true_value'] = boolean_results.apply(
                    lambda x: x['boolean_predicted'] if x['boolean_match'] else not x['boolean_predicted'],
                    axis=1
                )
                
                # Drop rows with missing values and ensure both columns have boolean type
                cm_data = boolean_results.dropna(subset=['true_value', 'boolean_predicted'])
                # Explicitly convert to bool type to handle any non-boolean values
                cm_data['true_value'] = cm_data['true_value'].astype(bool)
                cm_data['boolean_predicted'] = cm_data['boolean_predicted'].astype(bool)
                
                if len(cm_data) > 0:
                    # Create confusion matrix                  
                    cm = confusion_matrix(cm_data['true_value'], cm_data['boolean_predicted'], labels=[True, False])
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Plot
                    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
                              xticklabels=[True, False], yticklabels=[True, False], ax=axes[0, 1])
                    
                    axes[0, 1].set_title('Boolean Prediction Confusion Matrix', fontsize=15)
                    axes[0, 1].set_xlabel('Predicted', fontsize=12)
                    axes[0, 1].set_ylabel('True', fontsize=12)
        
        # Plot 3: Overall confidence (bottom left)
        if 'boolean_confidence' in boolean_results:
            confidence_mean = boolean_results['boolean_confidence'].mean()
            confidence_std = boolean_results['boolean_confidence'].std()
            
            bar = axes[1, 0].bar(['Overall'], [confidence_mean], yerr=[confidence_std],
                                capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[1, 0].text(0, confidence_mean + 0.02, f'{confidence_mean:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[1, 0].set_title('Boolean Prediction Confidence', fontsize=15)
            axes[1, 0].set_ylabel('Confidence', fontsize=12)
            axes[1, 0].set_ylim(0, min(1.0, confidence_mean + 0.15))
        
        # Plot 4: Match rate by query type (bottom right)
        has_query_type_data = False
        if 'query_type' in boolean_results.columns:
            query_types = ['observation', 'counterfactual_match']
            type_results = boolean_results[boolean_results['query_type'].isin(query_types)]
            
            if len(type_results) > 0 and 'boolean_match' in type_results:
                # Calculate match rate by type
                match_by_type = type_results.groupby('query_type')['boolean_match'].agg(['mean', 'std']).reset_index()
                
                if not match_by_type.empty:
                    has_query_type_data = True
                    # Plot bar chart
                    bars = axes[1, 1].bar(match_by_type['query_type'], match_by_type['mean'], 
                                        yerr=match_by_type['std'], capsize=5,
                                        color=sns.color_palette('viridis', len(match_by_type)))
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                      f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
                    axes[1, 1].set_title('Boolean Match Rate by Query Type', fontsize=15)
                    axes[1, 1].set_xlabel('Query Type', fontsize=12)
                    axes[1, 1].set_ylabel('Match Rate', fontsize=12)
                    axes[1, 1].set_ylim(0, 1.0)
        
        if not has_query_type_data:
            # Remove empty subplot if no query_type data
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        
        # Save combined plot
        output_path = os.path.join(vis_dir, 'boolean_metrics_overview.png')
        plt.savefig(output_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        outputs['boolean_metrics_overview'] = output_path
        
        return outputs
    
    def plot_directional_metrics(self, results_df, output_dir, show_plots=False):
        """Plot directional metrics: match rates, confusion matrix, and sentiment analysis in a consolidated view."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'directional')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Filter for directional evaluations
        directional_results = results_df[results_df["is_directional"] == True].copy()
        if len(directional_results) == 0:
            return outputs
        
        # Create a figure with subplots: 2 rows, 3 columns 
        # To accommodate the sentiment analysis visualization
        fig, axes = self._create_subplot_figure(2, 3, figsize=(18, 12))
        
        # Plot 1: Overall match rate (top left)
        if 'directional_match' in directional_results:
            match_rate = directional_results['directional_match'].mean()
            match_std = directional_results['directional_match'].std()
            
            bar = axes[0, 0].bar(['Overall'], [match_rate], yerr=[match_std], 
                               capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[0, 0].text(0, match_rate + 0.02, f'{match_rate:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[0, 0].set_title('Directional Match Rate', fontsize=15)
            axes[0, 0].set_ylabel('Match Rate', fontsize=12)
            axes[0, 0].set_ylim(0, min(1.0, match_rate + 0.15))  # Leave room for error bar and label
            
            # Plot 2: Confusion matrix for direction (top center)
            if 'directional_output_direction' in directional_results and 'directional_truth_direction' in directional_results:
                # Get only rows with valid directions
                directions = ['positive', 'negative', 'neutral', 'mixed']
                cm_data = directional_results[
                    directional_results['directional_output_direction'].isin(directions) &
                    directional_results['directional_truth_direction'].isin(directions)
                ]
                
                if len(cm_data) > 0:
                    # Create confusion matrix
                    cm = confusion_matrix(cm_data['directional_truth_direction'], 
                                        cm_data['directional_output_direction'],
                                        labels=directions)
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    
                    # Plot
                    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='viridis',
                              xticklabels=directions, yticklabels=directions, ax=axes[0, 1])
                    
                    axes[0, 1].set_title('Direction Prediction Confusion Matrix', fontsize=15)
                    axes[0, 1].set_xlabel('Predicted', fontsize=12)
                    axes[0, 1].set_ylabel('True', fontsize=12)
                    
            # Plot 3: Sentiment Analysis Results (top right)
            if 'directional_sentiment_compound' in directional_results:
                # Group sentiment scores into categories for visualization
                def categorize_sentiment(row):
                    if row['directional_sentiment_compound'] >= 0.05:
                        return 'Positive'
                    elif row['directional_sentiment_compound'] <= -0.05:
                        return 'Negative'
                    else:
                        return 'Neutral'
                
                directional_results['sentiment_category'] = directional_results.apply(categorize_sentiment, axis=1)
                sentiment_counts = directional_results['sentiment_category'].value_counts()
                
                # Plot pie chart of sentiment distribution
                if not sentiment_counts.empty:
                    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
                    sentiment_colors = [colors.get(cat) for cat in sentiment_counts.index]
                    
                    axes[0, 2].pie(sentiment_counts, labels=sentiment_counts.index, 
                                 autopct='%1.1f%%', startangle=90, colors=sentiment_colors)
                    axes[0, 2].set_title('Sentiment Distribution', fontsize=15)
                    
                    # Add average sentiment compound score as text
                    avg_sentiment = directional_results['directional_sentiment_compound'].mean()
                    axes[0, 2].text(0, -1.2, f'Avg. Sentiment: {avg_sentiment:.2f}', 
                                  ha='center', fontsize=12)
        
        # Plot 4: Overall confidence (bottom left)
        if 'directional_confidence' in directional_results:
            confidence_mean = directional_results['directional_confidence'].mean()
            confidence_std = directional_results['directional_confidence'].std()
            
            bar = axes[1, 0].bar(['Overall'], [confidence_mean], yerr=[confidence_std],
                               capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[1, 0].text(0, confidence_mean + 0.02, f'{confidence_mean:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[1, 0].set_title('Directional Prediction Confidence', fontsize=15)
            axes[1, 0].set_ylabel('Confidence', fontsize=12)
            axes[1, 0].set_ylim(0, min(1.0, confidence_mean + 0.15))
        
        # Plot 5: Match rate by query type (bottom center)
        has_query_type_data = False
        if 'query_type' in directional_results.columns:
            query_types = ['observation', 'counterfactual_match']
            type_results = directional_results[directional_results['query_type'].isin(query_types)]
            
            if len(type_results) > 0 and 'directional_match' in type_results:
                # Calculate match rate by type
                match_by_type = type_results.groupby('query_type')['directional_match'].agg(['mean', 'std']).reset_index()
                
                if not match_by_type.empty:
                    has_query_type_data = True
                    # Plot bar chart
                    bars = axes[1, 1].bar(match_by_type['query_type'], match_by_type['mean'], 
                                        yerr=match_by_type['std'], capsize=5,
                                        color=sns.color_palette('viridis', len(match_by_type)))
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                      f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
                    axes[1, 1].set_title('Directional Match Rate by Query Type', fontsize=15)
                    axes[1, 1].set_xlabel('Query Type', fontsize=12)
                    axes[1, 1].set_ylabel('Match Rate', fontsize=12)
                    axes[1, 1].set_ylim(0, 1.0)
        
        # Plot 6: Keyword vs Sentiment Direction Agreement (bottom right)
        if 'directional_keyword_direction' in directional_results and 'directional_sentiment_direction' in directional_results:
            # Compare keyword direction and sentiment direction
            directional_results['methods_agree'] = (
                directional_results['directional_keyword_direction'] == directional_results['directional_sentiment_direction']
            )
            
            agreement_rate = directional_results['methods_agree'].mean()
            
            # Create dictionary for agreement by direction
            agreement_by_dir = {}
            for direction in ['positive', 'negative', 'neutral', 'mixed']:
                dir_subset = directional_results[directional_results['directional_keyword_direction'] == direction]
                if len(dir_subset) > 0:
                    agreement_by_dir[direction] = dir_subset['methods_agree'].mean()
            
            if agreement_by_dir:
                # Plot
                bars = axes[1, 2].bar(agreement_by_dir.keys(), agreement_by_dir.values(),
                                    color=sns.color_palette('viridis', len(agreement_by_dir)))
                
                # Add value labels
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                  f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                
                axes[1, 2].set_title('Keyword-Sentiment Agreement by Direction', fontsize=15)
                axes[1, 2].set_xlabel('Direction (Keyword Method)', fontsize=12)
                axes[1, 2].set_ylabel('Agreement Rate', fontsize=12)
                axes[1, 2].set_ylim(0, 1.0)
                
                # Add overall agreement as text
                axes[1, 2].text(len(agreement_by_dir)/2 - 0.5, -0.1, f'Overall: {agreement_rate:.2f}',
                              ha='center', fontsize=12)
        
        # Remove any empty subplots - safer method to check if subplot is empty
        for row in range(2):
            for col in range(3):
                ax = axes[row, col]
                # Check different types of content that might be present
                has_content = (
                    len(ax.get_lines()) > 0 or          # Line plots
                    len(ax.patches) > 0 or              # Bar plots, pie charts
                    len(ax.collections) > 0 or          # Heatmaps, scatter plots
                    len(ax.texts) > 0 or                # Text annotations
                    len(ax.get_images()) > 0            # Images like heatmaps
                )
                if not has_content:
                    fig.delaxes(ax)
        
        plt.tight_layout()
        
        # Save combined plot
        output_path = os.path.join(vis_dir, 'directional_metrics_overview.png')
        plt.savefig(output_path, dpi=300)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        outputs['directional_metrics_overview'] = output_path
        
        return outputs
    
    def plot_text_metrics(self, results_df, output_dir, show_plots=False):
        """Plot text metrics: BLEU scores and semantic similarity in a consolidated view."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'text')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Create a figure with 2x2 subplots grid
        fig, axes = self._create_subplot_figure(2, 2, figsize=(14, 12))
        
        has_bleu = False
        has_sem = False
        has_bleu_by_type = False
        has_sem_by_type = False
        
        # Plot 1: Overall BLEU score (top left)
        if 'text_bleu_score' in results_df:
            has_bleu = True
            bleu_mean = results_df['text_bleu_score'].mean()
            bleu_std = results_df['text_bleu_score'].std()
            
            bar = axes[0, 0].bar(['Overall'], [bleu_mean], yerr=[bleu_std], 
                               capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[0, 0].text(0, bleu_mean + 0.02, f'{bleu_mean:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[0, 0].set_title('Average BLEU Score', fontsize=15)
            axes[0, 0].set_ylabel('BLEU Score', fontsize=12)
            axes[0, 0].set_ylim(0, min(1.0, bleu_mean + 0.15))  # Leave room for error bar and label
        
        # Plot 2: Overall semantic similarity (top right)
        if 'text_semantic_similarity' in results_df:
            has_sem = True
            sem_mean = results_df['text_semantic_similarity'].mean()
            sem_std = results_df['text_semantic_similarity'].std()
            
            bar = axes[0, 1].bar(['Overall'], [sem_mean], yerr=[sem_std], 
                               capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[0, 1].text(0, sem_mean + 0.02, f'{sem_mean:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[0, 1].set_title('Average Semantic Similarity', fontsize=15)
            axes[0, 1].set_ylabel('Similarity Score', fontsize=12)
            axes[0, 1].set_ylim(0, min(1.0, sem_mean + 0.15))
        
        # By query type if available
        if 'query_type' in results_df.columns:
            query_types = ['observation', 'counterfactual_match']
            type_results = results_df[results_df['query_type'].isin(query_types)]
            
            if len(type_results) > 0:
                # Plot 3: BLEU by query type (bottom left)
                if 'text_bleu_score' in type_results:
                    bleu_by_type = type_results.groupby('query_type')['text_bleu_score'].agg(['mean', 'std']).reset_index()
                    
                    if not bleu_by_type.empty:
                        has_bleu_by_type = True
                        bars = axes[1, 0].bar(bleu_by_type['query_type'], bleu_by_type['mean'], 
                                            yerr=bleu_by_type['std'], capsize=5,
                                            color=sns.color_palette('viridis', len(bleu_by_type)))
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                          f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                        
                        axes[1, 0].set_title('BLEU Score by Query Type', fontsize=15)
                        axes[1, 0].set_xlabel('Query Type', fontsize=12)
                        axes[1, 0].set_ylabel('BLEU Score', fontsize=12)
                        axes[1, 0].set_ylim(0, 1.0)
                
                # Plot 4: Semantic similarity by query type (bottom right)
                if 'text_semantic_similarity' in type_results:
                    sem_by_type = type_results.groupby('query_type')['text_semantic_similarity'].agg(['mean', 'std']).reset_index()
                    
                    if not sem_by_type.empty:
                        has_sem_by_type = True
                        bars = axes[1, 1].bar(sem_by_type['query_type'], sem_by_type['mean'], 
                                            yerr=sem_by_type['std'], capsize=5,
                                            color=sns.color_palette('viridis', len(sem_by_type)))
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                          f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                        
                        axes[1, 1].set_title('Semantic Similarity by Query Type', fontsize=15)
                        axes[1, 1].set_xlabel('Query Type', fontsize=12)
                        axes[1, 1].set_ylabel('Similarity Score', fontsize=12)
                        axes[1, 1].set_ylim(0, 1.0)
        
        # Remove empty subplots
        if not has_bleu:
            fig.delaxes(axes[0, 0])
        if not has_sem:
            fig.delaxes(axes[0, 1])
        if not has_bleu_by_type:
            fig.delaxes(axes[1, 0])
        if not has_sem_by_type:
            fig.delaxes(axes[1, 1])
        
        # Only save if at least one subplot exists
        if has_bleu or has_sem or has_bleu_by_type or has_sem_by_type:
            plt.tight_layout()
            
            # Save combined plot
            output_path = os.path.join(vis_dir, 'text_metrics_overview.png')
            plt.savefig(output_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            outputs['text_metrics_overview'] = output_path
        
        return outputs
    
    def plot_financial_metrics(self, results_df, output_dir, show_plots=False):
        """Plot financial metrics: asset match rates in a consolidated view."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'financial')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Filter for financial evaluations
        financial_results = results_df[results_df["is_financial"] == True].copy()
        if len(financial_results) == 0:
            return outputs
        
        # Create a figure with 1x2 subplots grid
        fig, axes = self._create_subplot_figure(1, 2, figsize=(14, 6))
        
        # Plot 1: Overall match rate (left)
        if 'financial_assets_match' in financial_results:
            match_mean = financial_results['financial_assets_match'].mean()
            match_std = financial_results['financial_assets_match'].std()
            
            bar = axes[0, 0].bar(['Overall'], [match_mean], yerr=[match_std], 
                               capsize=5, color=sns.color_palette('viridis', 1))
            
            # Add value label
            axes[0, 0].text(0, match_mean + 0.02, f'{match_mean:.2f}', 
                          ha='center', va='bottom', fontsize=12)
            
            axes[0, 0].set_title('Financial Asset Match Rate', fontsize=15)
            axes[0, 0].set_ylabel('Match Rate', fontsize=12)
            axes[0, 0].set_ylim(0, min(1.0, match_mean + 0.15))  # Leave room for error bar and label
        
            # Plot 2: Match rate by query type (right)
            has_query_type_data = False
            if 'query_type' in financial_results.columns:
                query_types = ['observation', 'counterfactual_match']
                type_results = financial_results[financial_results['query_type'].isin(query_types)]
                
                if len(type_results) > 0 and 'financial_assets_match' in type_results:
                    match_by_type = type_results.groupby('query_type')['financial_assets_match'].agg(['mean', 'std']).reset_index()
                    
                    if not match_by_type.empty:
                        has_query_type_data = True
                        bars = axes[0, 1].bar(match_by_type['query_type'], match_by_type['mean'], 
                                            yerr=match_by_type['std'], capsize=5,
                                            color=sns.color_palette('viridis', len(match_by_type)))
                        
                        # Add value labels
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                          f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                        
                        axes[0, 1].set_title('Financial Asset Match Rate by Query Type', fontsize=15)
                        axes[0, 1].set_xlabel('Query Type', fontsize=12)
                        axes[0, 1].set_ylabel('Match Rate', fontsize=12)
                        axes[0, 1].set_ylim(0, 1.0)
            
            if not has_query_type_data:
                # Remove empty subplot if no query_type data
                fig.delaxes(axes[0, 1])
            
            plt.tight_layout()
            
            # Save combined plot
            output_path = os.path.join(vis_dir, 'financial_metrics_overview.png')
            plt.savefig(output_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            outputs['financial_metrics_overview'] = output_path
        
        return outputs
    
    def plot_counterfactual_metrics(self, results_df, output_dir, show_plots=False):
        """Plot counterfactual metrics: sign match and error distributions in a consolidated view."""
        vis_dir = os.path.join(output_dir, 'visualizations', 'counterfactual')
        os.makedirs(vis_dir, exist_ok=True)
        outputs = {}
        
        # Filter for counterfactual results (where counterfactual_direction_match exists)
        cf_results = results_df[
            results_df["counterfactual_direction_match"].notna() |
            results_df["counterfactual_change_sign_match"].notna()
        ].copy() if "counterfactual_direction_match" in results_df.columns else pd.DataFrame()
        
        if len(cf_results) == 0:
            return outputs
        
        # Create a figure with 2x2 subplots grid
        fig, axes = self._create_subplot_figure(2, 2, figsize=(14, 12))
        
        # Plot 1: Sign match rate (top left)
        has_sign_match = False
        if 'counterfactual_change_sign_match' in cf_results:
            valid_data = cf_results.dropna(subset=['counterfactual_change_sign_match'])
            
            if len(valid_data) > 0:
                has_sign_match = True
                sign_mean = valid_data['counterfactual_change_sign_match'].mean()
                sign_std = valid_data['counterfactual_change_sign_match'].std()
                
                bar = axes[0, 0].bar(['Overall'], [sign_mean], yerr=[sign_std], 
                                   capsize=5, color=sns.color_palette('viridis', 1))
                
                # Add value label
                axes[0, 0].text(0, sign_mean + 0.02, f'{sign_mean:.2f}', 
                              ha='center', va='bottom', fontsize=12)
                
                axes[0, 0].set_title('Counterfactual Sign Match Rate', fontsize=15)
                axes[0, 0].set_ylabel('Match Rate', fontsize=12)
                axes[0, 0].set_ylim(0, min(1.0, sign_mean + 0.15))  # Leave room for error bar and label
        
        # Plot 2: Sign match by query type (top right)
        has_sign_match_by_type = False
        if 'query_type' in cf_results.columns and 'counterfactual_change_sign_match' in cf_results:
            valid_data = cf_results.dropna(subset=['counterfactual_change_sign_match'])
            query_types = ['counterfactual_match']  # Typically only this type for counterfactuals
            type_results = valid_data[valid_data['query_type'].isin(query_types)]
            
            if len(type_results) > 0:
                sign_by_type = type_results.groupby('query_type')['counterfactual_change_sign_match'].agg(['mean', 'std']).reset_index()
                
                if not sign_by_type.empty:
                    has_sign_match_by_type = True
                    bars = axes[0, 1].bar(sign_by_type['query_type'], sign_by_type['mean'], 
                                        yerr=sign_by_type['std'], capsize=5,
                                        color=sns.color_palette('viridis', len(sign_by_type)))
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                      f'{height:.2f}', ha='center', va='bottom', fontsize=12)
                    
                    axes[0, 1].set_title('Sign Match Rate by Query Type', fontsize=15)
                    axes[0, 1].set_xlabel('Query Type', fontsize=12)
                    axes[0, 1].set_ylabel('Match Rate', fontsize=12)
                    axes[0, 1].set_ylim(0, 1.0)
        
        # Error violin plots in bottom row
        error_cols = ['counterfactual_change_magnitude_error', 'counterfactual_change_pct_error']
        has_magnitude_error = False
        has_pct_error = False
        
        for i, col in enumerate(error_cols):
            if col in cf_results:
                # Filter out infinite values and NaNs
                valid_data = cf_results[(cf_results[col] < float('inf')) & cf_results[col].notna()]
                
                if len(valid_data) > 0:
                    if i == 0:  # magnitude error (bottom left)
                        has_magnitude_error = True
                        # Clip at 95th percentile for better visualization
                        clip_val = min(valid_data[col].quantile(0.95), 5)
                        
                        # Create violin plot
                        plot_data = valid_data.copy()
                        plot_data['clipped_error'] = plot_data[col].clip(0, clip_val)
                        
                        sns.violinplot(data=plot_data, y='clipped_error', ax=axes[1, 0])
                        axes[1, 0].set_title('Distribution of Change Magnitude Error', fontsize=15)
                        axes[1, 0].set_ylabel('Error (clipped)', fontsize=12)
                        
                    elif i == 1:  # percent error (bottom right)
                        has_pct_error = True
                        # Clip at 95th percentile for better visualization
                        clip_val = min(valid_data[col].quantile(0.95), 5)
                        
                        # Create violin plot
                        plot_data = valid_data.copy()
                        plot_data['clipped_error'] = plot_data[col].clip(0, clip_val)
                        
                        sns.violinplot(data=plot_data, y='clipped_error', ax=axes[1, 1], cut=0)
                        axes[1, 1].set_title('Distribution of Change Percent Error', fontsize=15)
                        axes[1, 1].set_ylabel('Error (clipped)', fontsize=12)
        
        # Remove empty subplots
        if not has_sign_match:
            fig.delaxes(axes[0, 0])
        if not has_sign_match_by_type:
            fig.delaxes(axes[0, 1])
        if not has_magnitude_error:
            fig.delaxes(axes[1, 0])
        if not has_pct_error:
            fig.delaxes(axes[1, 1])
        
        # Only save if at least one subplot exists
        if has_sign_match or has_sign_match_by_type or has_magnitude_error or has_pct_error:
            plt.tight_layout()
            
            # Save combined plot
            output_path = os.path.join(vis_dir, 'counterfactual_metrics_overview.png')
            plt.savefig(output_path, dpi=300)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
                
            outputs['counterfactual_metrics_overview'] = output_path
        
        return outputs
        
    def create_visualizations(self, results_df: pd.DataFrame, output_dir: str, show_plots: bool = False):
        """
        Create basic visualizations for the evaluation results.
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Directory to save plots
            show_plots: Whether to display plots in addition to saving them
            
        Returns:
            Dictionary with paths to all created visualizations
        """
        visualization_paths = {}
        
        # Create main visualization directory
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Basic overview plots
        visualization_paths['types_distribution'] = self.plot_evaluation_types(results_df, output_dir, show_plots)
        visualization_paths['match_rates'] = self.plot_match_rates(results_df, output_dir, show_plots)
        visualization_paths['performance_overview'] = self.plot_performance_overview(results_df, output_dir, show_plots)
        
        # Detailed metric plots
        visualization_paths['numerical'] = self.plot_numerical_metrics(results_df, output_dir, show_plots)
        visualization_paths['boolean'] = self.plot_boolean_metrics(results_df, output_dir, show_plots)
        visualization_paths['directional'] = self.plot_directional_metrics(results_df, output_dir, show_plots)
        visualization_paths['text'] = self.plot_text_metrics(results_df, output_dir, show_plots)
        visualization_paths['financial'] = self.plot_financial_metrics(results_df, output_dir, show_plots)
        visualization_paths['counterfactual'] = self.plot_counterfactual_metrics(results_df, output_dir, show_plots)
        
        return visualization_paths



__all__ = ['LLMOutputEvaluator']