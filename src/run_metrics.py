import os
import argparse
import json
import pandas as pd
import datetime
from typing import Optional

from causal_world_modelling_agent.utils.metric_utils import LLMOutputEvaluator


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
    parser.add_argument("--observations_col", type=str, default="observations",
                      help="Column name containing target observation values (for counterfactual evaluation)")
    parser.add_argument("--query_type_col", type=str, default="type",
                      help="Column name containing query type information")
    parser.add_argument("--target_col", type=str, default="target_variable",
                      help="Column name containing target variable name information")
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
         observations_col: Optional[str] = None,
         query_type_col: Optional[str] = None,
         target_col: Optional[str] = None,
         no_semantic: bool = False,
         semantic_model: str = "all-mpnet-base-v2",
         show_plots: bool = False) -> None:
    """
    Main function to run the LLM output evaluation.
    
    Args:
        input_csv: Path to input CSV file with outputs and ground truths
        output_dir: Directory to save results and visualizations
        llm_output_col: Column name containing LLM outputs
        ground_truth_col: Column name containing ground truth values
        observations_col: Column name containing target observations (optional)
        query_type_col: Column name containing query type (optional)
        no_semantic: Whether to disable semantic similarity computation
        semantic_model: Model to use for semantic similarity
        show_plots: Whether to display plots in addition to saving them
    """
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Create evaluator
    evaluator = LLMOutputEvaluator(use_semantic=not no_semantic, semantic_model=None if no_semantic else semantic_model)
    
    # Evaluate
    print("Evaluating LLM outputs...")
    results_df = evaluator.batch_evaluate(
        df,
        llm_output_col=llm_output_col, 
        ground_truth_col=ground_truth_col,
        observations_col=observations_col,
        query_type_col=query_type_col,
        target_col=target_col,
    )
    
    # Analyze results
    print("Analyzing results...")
    analysis = evaluator.summarize_results(results_df)
      # Print high-level summary
    print("\nQuick Summary:")
    if 'overall_performance' in analysis:
        print(f"Overall Performance: {analysis['overall_performance']*100:.1f}%")
    
    # Print performance by type
    if 'match_rates' in analysis:
        print("\nPerformance by type:")
        for eval_type, metrics in analysis['match_rates'].items():
            print(f"  {eval_type.title()}:")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"    - {metric_name}: {value*100:.1f}%")
        
    # Count types
    type_counts = {}
    for col in results_df.columns:
        if col.endswith('_applicable'):
            eval_type = col.split('_')[0]
            applicable_count = results_df[col].sum()
            if applicable_count > 0:
                type_counts[eval_type] = applicable_count
                
    print(f"\nEvaluation types applied: {', '.join(type_counts.keys())}")
    
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
      # Create all visualizations
    print(f"Creating visualizations...")
    visualization_paths = evaluator.create_visualizations(results_df, output_dir, show_plots)
    
    # Log what was created
    vis_created = [k for k, v in visualization_paths.items() if v is not None]
    print(f"Created visualizations: {', '.join(vis_created)}")
    print(f"All visualizations saved to {os.path.join(output_dir, 'visualizations')}")
      # Print evaluation types count
    if 'type_counts' in analysis:
        print("\nEvaluation type counts:")
        for eval_type, count in analysis['type_counts'].items():
            print(f"  {eval_type.title()}: {count}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))