#!/usr/bin/env python
import os
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

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
    document_stats = defaultdict(lambda: {"total": 0, "correct": 0, "confidence_sum": 0})
    section_stats = defaultdict(lambda: {"total": 0, "correct": 0, "confidence_sum": 0})

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
            
            # Analyze document and section statistics
            for _, row in df.iterrows():
                doc_name = row['document_name']
                section_context = row['section_context']
                
                # Update document statistics
                document_stats[doc_name]["total"] += 1
                if row['current_name'] == row['proposed_name']:
                    document_stats[doc_name]["correct"] += 1
                document_stats[doc_name]["confidence_sum"] += row['confidence']
                
                # Update section statistics if section context exists
                if pd.notna(section_context) and section_context.strip():
                    sections = section_context.split("; ")
                    for section in sections:
                        section_stats[section]["total"] += 1
                        if row['current_name'] == row['proposed_name']:
                            section_stats[section]["correct"] += 1
                        section_stats[section]["confidence_sum"] += row['confidence']
            
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
    
    # Print document statistics
    print("\n=== Document Statistics ===")
    doc_stats_list = []
    for doc_name, stats in document_stats.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"] * 100)
            avg_conf = stats["confidence_sum"] / stats["total"]
            doc_stats_list.append({
                "Document": doc_name,
                "Total Fields": stats["total"],
                "Correct Fields": stats["correct"],
                "Accuracy (%)": round(accuracy, 2),
                "Avg Confidence": round(avg_conf, 4)
            })
    
    doc_stats_df = pd.DataFrame(doc_stats_list)
    doc_stats_df = doc_stats_df.sort_values("Accuracy (%)", ascending=False)
    print(doc_stats_df.to_string(index=False))
    
    # Print section statistics
    print("\n=== Section Statistics ===")
    section_stats_list = []
    for section, stats in section_stats.items():
        if stats["total"] > 0:
            accuracy = (stats["correct"] / stats["total"] * 100)
            avg_conf = stats["confidence_sum"] / stats["total"]
            section_stats_list.append({
                "Section": section,
                "Total Fields": stats["total"],
                "Correct Fields": stats["correct"],
                "Accuracy (%)": round(accuracy, 2),
                "Avg Confidence": round(avg_conf, 4)
            })
    
    section_stats_df = pd.DataFrame(section_stats_list)
    section_stats_df = section_stats_df.sort_values("Accuracy (%)", ascending=False)
    print(section_stats_df.to_string(index=False))

    # Save detailed results to CSV files
    results_df.to_csv("report_analysis_summary.csv", index=False)
    doc_stats_df.to_csv("document_statistics.csv", index=False)
    section_stats_df.to_csv("section_statistics.csv", index=False)
    
    print("\nDetailed results have been saved to:")
    print("- report_analysis_summary.csv")
    print("- document_statistics.csv")
    print("- section_statistics.csv")

if __name__ == "__main__":
    analyze_reports() 