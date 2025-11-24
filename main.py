"""
Main script for Swahili Text Data Pipeline
Orchestrates data ingestion, parsing, and inspection.
"""

import argparse
import json
import os
from datetime import datetime
from data_pipeline import DataPipeline
from data_inspector import DataInspector


def save_report(stats: dict, output_path: str):
    """
    Save statistics report to JSON file.
    
    Args:
        stats: Statistics dictionary
        output_path: Path to save the report
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Swahili Text Data Pipeline - Data Engineering and Preprocessing"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing data files (default: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Directory to save reports (default: reports)'
    )
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=['train.txt', 'test.txt', 'valid.txt'],
        help='List of data files to process (default: train.txt test.txt valid.txt)'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.save_report:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline and inspector
    pipeline = DataPipeline(data_dir=args.data_dir)
    inspector = DataInspector()
    
    # Process each dataset
    all_results = {}
    
    for filename in args.files:
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print(f"{'='*80}")
        
        try:
            # Get basic file info
            file_info = pipeline.get_dataset_info(filename)
            print(f"\nFile Information:")
            for key, value in file_info.items():
                if key != "error":
                    print(f"  {key}: {value}")
            
            if "error" in file_info:
                print(f"  ERROR: {file_info['error']}")
                continue
            
            # Parse dataset
            features, labels, label_format = pipeline.parse_dataset(filename)
            
            if not features:
                print(f"  WARNING: No data found in {filename}")
                continue
            
            # Inspect dataset
            stats = inspector.inspect_dataset(features, labels if any(l is not None for l in labels) else None)
            
            # Add metadata
            stats["metadata"] = {
                "filename": filename,
                "label_format": label_format,
                "processing_date": datetime.now().isoformat(),
                "file_info": file_info
            }
            
            # Print summary
            dataset_name = filename.replace('.txt', '').upper()
            inspector.print_summary(stats, dataset_name)
            
            # Store results
            all_results[filename] = stats
            
            # Save individual report if requested
            if args.save_report:
                report_path = os.path.join(
                    args.output_dir,
                    f"{filename.replace('.txt', '')}_report.json"
                )
                save_report(stats, report_path)
        
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
        except Exception as e:
            print(f"  ERROR processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined report if requested
    if args.save_report and all_results:
        combined_report_path = os.path.join(args.output_dir, "combined_report.json")
        combined_report = {
            "processing_date": datetime.now().isoformat(),
            "datasets": all_results,
            "summary": {
                "total_datasets": len(all_results),
                "dataset_names": list(all_results.keys())
            }
        }
        save_report(combined_report, combined_report_path)
    
    print(f"\n{'='*80}")
    print("Data Pipeline Processing Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
