"""
Main module for geomapr package
"""

import argparse
import sys
import os
from .metadata_processor import MetadataProcessor
from .metadata_mapper import MetadataMapper




def main():
    """Main entry point with subcommands"""
    parser = argparse.ArgumentParser(
        description="GeoMapr: Retrieve and map metadata from GEO/SRA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  retrieve    Retrieve metadata from GEO/SRA
  map         Map input CSV to target template using AI  
  pipeline    Complete pipeline: retrieve + map

Examples:
  geomapr retrieve GSE212964
  geomapr map input.csv template.csv output.csv
  geomapr pipeline GSE212964 template.csv
        """
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Retrieve subcommand
    retrieve_parser = subparsers.add_parser(
        'retrieve',
        help='Retrieve metadata from GEO/SRA',
        description="Retrieve and transform metadata from GEO/SRA for data repositories"
    )
    retrieve_parser.add_argument(
        "geo_series",
        help="GEO series identifier (e.g., GSE212963)"
    )
    retrieve_parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: {geo_series}_metadata.csv)"
    )
    retrieve_parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only show series summary, don't process all files"
    )
    
    # Map subcommand  
    map_parser = subparsers.add_parser(
        'map',
        help='Map input CSV to target template using AI',
        description="Map input CSV to target CSV template using AI and NF metadata dictionary"
    )
    map_parser.add_argument(
        "input_csv",
        help="Input CSV file with metadata to map from"
    )
    map_parser.add_argument(
        "target_csv",
        help="Target CSV template file to fill in"
    )
    map_parser.add_argument(
        "output_csv",
        help="Output CSV/Excel file path"
    )
    map_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )
    map_parser.add_argument(
        "--creds",
        default="creds.yaml",
        help="Credentials file path (default: creds.yaml)"
    )
    
    # Pipeline subcommand
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Complete pipeline: retrieve + map',
        description="Complete pipeline: Retrieve GEO/SRA metadata and map to target CSV template using AI"
    )
    pipeline_parser.add_argument(
        "geo_series",
        help="GEO series identifier (e.g., GSE212963)"
    )
    pipeline_parser.add_argument(
        "target_csv",
        help="Target CSV template file to fill in"
    )
    pipeline_parser.add_argument(
        "-o", "--output",
        help="Output file path (default: {geo_series}_mapped.xlsx)"
    )
    pipeline_parser.add_argument(
        "--intermediate-csv",
        help="Save intermediate GEO/SRA metadata to this file (default: {geo_series}_metadata.csv)"
    )
    pipeline_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml)"
    )
    pipeline_parser.add_argument(
        "--creds",
        default="creds.yaml",
        help="Credentials file path (default: creds.yaml)"
    )
    pipeline_parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the intermediate GEO/SRA metadata CSV file"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'retrieve':
        retrieve_command_impl(args)
    elif args.command == 'map':
        map_command_impl(args)
    elif args.command == 'pipeline':
        pipeline_command_impl(args)
    else:
        parser.print_help()
        sys.exit(1)


def retrieve_command_impl(args):
    """Implementation of retrieve command"""
    # Initialize processor
    processor = MetadataProcessor()
    
    try:
        if args.summary_only:
            # Get summary only
            print(f"Getting summary for {args.geo_series}...")
            summary = processor.get_series_summary(args.geo_series)
            
            if 'error' in summary:
                print(f"Error: {summary['error']}")
                sys.exit(1)
            
            print("\n=== Series Summary ===")
            print(f"Series ID: {summary['series_id']}")
            print(f"Title: {summary['title']}")
            print(f"Organism: {summary['organism']}")
            print(f"Platform: {summary['platform']}")
            print(f"Submission Date: {summary['submission_date']}")
            print(f"SRA Project: {summary.get('sra_project', 'None')}")
            print(f"Has SRA Data: {summary.get('has_sra_data', False)}")
            print(f"Total SRA Runs: {summary.get('total_sra_runs', 0)}")
            print(f"Unique Samples: {summary.get('unique_samples', 0)}")
            print(f"Unique Experiments: {summary.get('unique_experiments', 0)}")
            
            if summary.get('organisms_in_sra'):
                print(f"Organisms in SRA: {', '.join(summary['organisms_in_sra'][:5])}")
            
            if summary.get('library_strategies'):
                print(f"Library Strategies: {', '.join(summary['library_strategies'][:5])}")
        
        else:
            # Process full series
            output_file = args.output
            if not output_file:
                output_file = f"{args.geo_series}_metadata.csv"
            
            result_df = processor.process_geo_series(args.geo_series, output_file)
            
            if result_df.empty:
                print("No data found for the specified series.")
                sys.exit(1)
            
            print(f"\nSuccess! Generated {len(result_df)} rows of metadata.")
            print(f"Output saved to: {output_file}")
            
            # Show a preview
            if len(result_df) > 0:
                print("\n=== Preview (first 3 files) ===")
                preview_cols = ['run_accession', 'file_name', 'file_type', 'file_size_mb', 'file_s3_location']
                available_cols = [col for col in preview_cols if col in result_df.columns]
                
                if not available_cols:
                    # Fallback to old column names if new ones don't exist
                    preview_cols = ['run_accession', 'sample_accession', 'file_name', 
                                   'organism_name', 'library_strategy']
                    available_cols = [col for col in preview_cols if col in result_df.columns]
                
                print(result_df[available_cols].head(3).to_string(index=False))
                
    except Exception as e:
        print(f"Error processing {args.geo_series}: {str(e)}")
        sys.exit(1)


def map_command_impl(args):
    """Implementation of map command"""
    try:
        mapper = MetadataMapper(config_path=args.config, creds_path=args.creds)
        mapper.process_csv_mapping(
            input_csv_path=args.input_csv,
            target_csv_path=args.target_csv,
            output_csv_path=args.output_csv
        )
    except Exception as e:
        print(f"❌ Mapping failed: {str(e)}")
        sys.exit(1)


def pipeline_command_impl(args):
    """Implementation of pipeline command"""
    # Set default file names
    output_file = args.output or f"{args.geo_series}_mapped.xlsx"
    intermediate_csv = args.intermediate_csv or f"{args.geo_series}_metadata.csv"
    
    try:
        print("🔍 STEP 1: Retrieving GEO/SRA metadata...")
        print(f"Processing series: {args.geo_series}")
        
        # Initialize metadata processor
        processor = MetadataProcessor()
        
        # Get GEO/SRA metadata
        result_df = processor.process_geo_series(args.geo_series, intermediate_csv)
        
        if result_df.empty:
            print("❌ No data found for the specified series.")
            sys.exit(1)
        
        print(f"✅ Retrieved metadata for {len(result_df)} files")
        print(f"   Intermediate data saved to: {intermediate_csv}")
        
        # Check if target CSV exists
        if not os.path.exists(args.target_csv):
            print(f"❌ Error: Target CSV file not found: {args.target_csv}")
            sys.exit(1)
        
        print(f"\n🤖 STEP 2: AI-powered mapping to target template...")
        print(f"Target template: {args.target_csv}")
        print(f"Using config: {args.config}")
        print(f"Using credentials: {args.creds}")
        
        # Initialize AI mapper
        mapper = MetadataMapper(config_path=args.config, creds_path=args.creds)
        
        # Process the mapping
        mapper.process_csv_mapping(
            input_csv_path=intermediate_csv,
            target_csv_path=args.target_csv,
            output_csv_path=output_file.replace('.xlsx', '.csv')
        )
        
        print(f"\n🎉 PIPELINE COMPLETE!")
        print(f"   Final results: {output_file}")
        print(f"   CSV version: {output_file.replace('.xlsx', '.csv')}")
        
        # Clean up intermediate file if requested
        if not args.keep_intermediate:
            try:
                os.remove(intermediate_csv)
                print(f"   Cleaned up intermediate file: {intermediate_csv}")
            except OSError:
                pass  # File might not exist or permission issues
        else:
            print(f"   Intermediate file kept: {intermediate_csv}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        # Clean up intermediate file on error unless explicitly keeping it
        if not args.keep_intermediate and os.path.exists(intermediate_csv):
            try:
                os.remove(intermediate_csv)
            except OSError:
                pass
        sys.exit(1)


if __name__ == "__main__":
    main()
