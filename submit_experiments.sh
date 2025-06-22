#!/bin/bash
"""
CausalHES Experiment Submission Script

This script provides an easy interface to submit CausalHES experiments
to the GPU cluster.

Usage:
    ./submit_experiments.sh [option]

Options:
    irish       - Run Irish dataset experiments
    process     - Process Irish dataset only
    help        - Show this help message

Examples:
    ./submit_experiments.sh irish       # Irish experiments
    ./submit_experiments.sh process     # Process data only
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "CausalHES Experiment Submission Script"
    echo ""
    echo "Usage: ./submit_experiments.sh [option]"
    echo ""
    echo "Options:"
    echo "  irish       - Run Irish dataset experiments"
    echo "  process     - Process Irish dataset only"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./submit_experiments.sh irish       # Irish experiments"
    echo "  ./submit_experiments.sh process     # Process data only"
    echo ""
    echo "Job Monitoring:"
    echo "  bjobs                    # Check job status"
    echo "  bjobs -l <job_id>       # Detailed job info"
    echo "  tail -f jobs/output_*   # Monitor output"
    echo "  bkill <job_id>          # Cancel job"
}

# Function to check if jobs directory exists
check_jobs_dir() {
    if [ ! -d "jobs" ]; then
        print_error "Jobs directory not found!"
        print_error "Please ensure you're in the CausalHES project directory."
        exit 1
    fi
}

# Function to submit job and show status
submit_job() {
    local job_script=$1
    local job_name=$2
    
    if [ ! -f "$job_script" ]; then
        print_error "Job script not found: $job_script"
        exit 1
    fi
    
    print_status "Submitting $job_name..."
    
    # Submit job and capture job ID
    job_output=$(bsub < "$job_script" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID from output
        job_id=$(echo "$job_output" | grep -o 'Job <[0-9]*>' | grep -o '[0-9]*')
        print_success "$job_name submitted successfully!"
        print_status "Job ID: $job_id"
        print_status "Monitor with: bjobs -l $job_id"
        print_status "View output: tail -f jobs/output_*_$job_id.out"
        echo ""
    else
        print_error "Failed to submit $job_name"
        echo "$job_output"
        exit 1
    fi
}

# Main script
main() {
    local option=${1:-help}
    
    # Check if we're in the right directory
    check_jobs_dir
    
    case $option in
        "irish")
            print_status "Submitting Irish dataset experiments..."
            
            # Check if Irish data is processed
            if [ ! -f "data/processed_irish/irish_dataset_processed.npz" ]; then
                print_warning "Processed Irish data not found."
                print_status "You may need to run: ./submit_experiments.sh process"
                print_status "Continuing anyway - the job will handle data processing if raw data exists."
            fi
            
            submit_job "jobs/run_irish_experiments.sh" "Irish Dataset Experiments"
            print_success "Irish dataset experiments submitted!"
            print_status "Estimated runtime: 3-4 hours"
            ;;
            
        "process")
            print_status "Submitting Irish data processing job..."
            
            # Check if raw Irish data exists
            if [ ! -d "data/Irish" ] || [ ! -f "data/Irish/ElectricityConsumption.csv" ]; then
                print_error "Raw Irish data not found!"
                print_error "Please ensure data/Irish/ contains:"
                print_error "  - ElectricityConsumption.csv"
                print_error "  - household_characteristics.csv"
                exit 1
            fi
            
            submit_job "jobs/process_irish_data.sh" "Irish Data Processing"
            print_success "Irish data processing submitted!"
            print_status "Estimated runtime: 30-60 minutes"
            ;;
            
        "help"|"-h"|"--help")
            show_help
            ;;
            
        *)
            print_error "Unknown option: $option"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
