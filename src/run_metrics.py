
import os
import argparse
import json
import pandas as pd
import datetime
from typing import Optional

from causal_world_modelling_agent.utils.metric_utils import LLMOutputEvaluator, visualize_evaluation_results



def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM outputs against ground truths.")
    parser.add_argument("--input_csv", type=str, required=True, 
                      help="Path to input CSV file with outputs and ground truths")
    parser.add_argument("--output_dir", type=str, default="../data/metric_results",
                      help="Directory to save results and visualizations")
    parser.add_argument("--llm_output_col", type=str, default="predicted_value",
                      help="Column name containing LLM outputs")
    parser.add_argument("--ground_truth_col", type=str, default="ground_truth",
                      help="Column name containing ground truth values")
    parser.add_argument("--no_semantic", action="store_true",
                      help="Disable semantic similarity computation")
    parser.add_argument("--semantic_model", type=str, default="all-mpnet-base-v2",
                      help="Model to use for semantic similarity computation")
    parser.add_argument("--show_plots", action="store_true",
                      help="Display plots in addition to saving them")
    
    return parser.parse_args()


def main(input_csv: str,
         output_dir: str,
         llm_output_col: str,
         ground_truth_col: str,
         no_semantic: bool,
         semantic_model: str,
         show_plots: bool) -> None:
    """
    Main function to run the LLM output evaluation.
    
    Args:
        input_csv: Path to input CSV file with outputs and ground truths
        output_dir: Directory to save results and visualizations
        llm_output_col: Column name containing LLM outputs
        ground_truth_col: Column name containing ground truth values
        no_semantic: Whether to disable semantic similarity computation
        show_plots: Whether to display plots in addition to saving them
    """
    
    # Load data
    df = pd.read_csv(input_csv)

    # Create evaluator
    evaluator = LLMOutputEvaluator(use_semantic=not no_semantic, semantic_model=None if no_semantic else semantic_model)
    
    # Evaluate
    print("Evaluating LLM outputs...")
    results_df = evaluator.batch_evaluate(
        df,
        llm_output_col, 
        ground_truth_col
    )
    
    # Analyze results
    print("Analyzing results...")
    analysis = evaluator.summarize_results(results_df)
    
    # Print analysis summary
    print("\nAnalysis Summary:")
    print(json.dumps(analysis, indent=2))
    
    # Create output directory
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_dir, current_time)
    os.makedirs(output_dir, exist_ok=True)

    # Save analysis to JSON
    analysis_file = os.path.join(output_dir, 'analysis_summary.json')
    with open(analysis_file, 'w') as f:
        json.dump(analysis, indent=2, fp=f)
    print(f"Analysis saved to {analysis_file}")
    
    # Save results DataFrame
    csv_path = os.path.join(output_dir, f'metric_results.csv')
    
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create visualizations
    visualize_evaluation_results(
        results_df, 
        os.path.join(output_dir, 'evaluation_results.png'),
        show_plot=show_plots
    )
    print(f"Visualizations saved to {output_dir}")
    
    # Print key metrics
    print("\nKey metrics:")
    for key, value in analysis.items():
        print(f"{key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))