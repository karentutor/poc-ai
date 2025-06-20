#!/usr/bin/env python
import os
import csv
import pandas as pd
from pathlib import Path

def analyze_reports():
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        print("No reports directory found!")
        return

    # Get all CSV files in the reports directory
    csv_files = list(Path(reports_dir).glob("*.csv"))
    if not csv_files:
        print("No report files found!")
        return

    # Create a list to store results
    results = []

    # Process each CSV file
    for csv_file in csv_files:
        pdf_name = csv_file.stem  # Get filename without extension
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Calculate accuracy (where current_name equals proposed_name)
            total_fields = len(df)
            correct_fields = sum(df['current_name'] == df['proposed_name'])
            accuracy = (correct_fields / total_fields * 100) if total_fields > 0 else 0
            
            # Get average confidence
            avg_confidence = df['confidence'].mean()
            
            # Get index usage statistics
            index_usage = df['index_used'].value_counts().to_dict()
            index_usage_str = ", ".join([f"{idx}: {count}" for idx, count in index_usage.items()])
            
            results.append({
                'PDF Name': pdf_name,
                'Accuracy (%)': round(accuracy, 2),
                'Total Fields': total_fields,
                'Correct Fields': correct_fields,
                'Avg Confidence': round(avg_confidence, 4),
                'Index Usage': index_usage_str
            })
            
        except Exception as e:
            print(f"Error processing {pdf_name}: {str(e)}")
            continue

    if not results:
        print("No valid reports could be processed!")
        return

    # Convert results to DataFrame and sort by PDF name and accuracy
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['PDF Name', 'Accuracy (%)'], ascending=[True, False])

    # Calculate overall statistics
    total_pdfs = len(results_df)
    avg_accuracy = results_df['Accuracy (%)'].mean()
    total_fields = results_df['Total Fields'].sum()
    total_correct = results_df['Correct Fields'].sum()
    overall_accuracy = (total_correct / total_fields * 100) if total_fields > 0 else 0

    # Print summary
    print("\n=== Report Analysis Summary ===")
    print(f"Total PDFs analyzed: {total_pdfs}")
    print(f"Average accuracy across all PDFs: {round(avg_accuracy, 2)}%")
    print(f"Overall accuracy (all fields): {round(overall_accuracy, 2)}%")
    print(f"Total fields processed: {total_fields}")
    print(f"Total correctly identified fields: {total_correct}")
    print("\nDetailed Results:")
    print(results_df.to_string(index=False))

    # Save detailed results to CSV
    output_file = "report_analysis_summary.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results have been saved to {output_file}")

if __name__ == "__main__":
    analyze_reports() 