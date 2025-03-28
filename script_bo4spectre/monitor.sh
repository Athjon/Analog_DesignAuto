#!/bin/bash
# Script for monitoring Bayesian optimization progress

# Default values
RESULTS_DIR=""
BROWSER="xdg-open"
# Update default search path to match your actual directory structure
DEFAULT_SEARCH_PATH="/home/jianghaoning/ICCAD/AnalogDesignAuto_MultiAgent/custom_env/run_test"

# Parse command line arguments
while getopts "d:b:p:" opt; do
  case $opt in
    d) RESULTS_DIR="$OPTARG"
       ;;
    b) BROWSER="$OPTARG"
       ;;
    p) DEFAULT_SEARCH_PATH="$OPTARG"
       ;;
    \?) echo "Invalid option: -$OPTARG" >&2
        echo "Usage: $0 [-d results_dir] [-b browser] [-p search_path]" >&2
        exit 1
        ;;
  esac
done

# If no results directory specified, search for the most recent one
if [ -z "$RESULTS_DIR" ]; then
    # Find the most recent run directory
    LATEST_DIR=$(find "$DEFAULT_SEARCH_PATH" -type d -name "run_*" | sort -r | head -n 1)
    
    if [ -z "$LATEST_DIR" ]; then
        echo "Error: No results directory found in $DEFAULT_SEARCH_PATH"
        echo "Check if the path is correct or specify a different search path with -p"
        exit 1
    fi
    
    RESULTS_DIR=$LATEST_DIR
    echo "Using latest results directory: $RESULTS_DIR"
fi

# Check if directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Directory does not exist: $RESULTS_DIR"
    exit 1
fi

# Check if HTML dashboard exists
DASHBOARD_PATH="$RESULTS_DIR/dashboard.html"
if [ ! -f "$DASHBOARD_PATH" ]; then
    echo "Warning: Dashboard file not found: $DASHBOARD_PATH"
    echo "Please wait for optimization to start, or check if directory is correct"
    
    # Try to find a different dashboard in the results directory
    ALT_DASHBOARD=$(find "$RESULTS_DIR" -name "dashboard.html" | head -n 1)
    if [ -n "$ALT_DASHBOARD" ]; then
        DASHBOARD_PATH="$ALT_DASHBOARD"
        echo "Found alternative dashboard: $DASHBOARD_PATH"
    else
        # Look for any HTML files in the directory
        HTML_FILES=$(find "$RESULTS_DIR" -name "*.html" | head -n 1)
        if [ -n "$HTML_FILES" ]; then
            DASHBOARD_PATH="$HTML_FILES"
            echo "Found HTML file: $DASHBOARD_PATH"
        else
            echo "No HTML files found. Will continue monitoring logs."
        fi
    fi
fi

# Open browser to view dashboard if it exists
if [ -f "$DASHBOARD_PATH" ]; then
    echo "Opening dashboard: $DASHBOARD_PATH"
    $BROWSER "$DASHBOARD_PATH"
fi

# Display real-time logs
LOG_FILE="$(find "$RESULTS_DIR" -name "*.log" | head -n 1)"
if [ -f "$LOG_FILE" ]; then
    echo "Displaying real-time log: $LOG_FILE"
    tail -f "$LOG_FILE"
else
    echo "Warning: No log file found in $RESULTS_DIR"
    echo "Searching for log files in subdirectories..."
    
    # Search deeper for log files
    LOG_FILE="$(find "$RESULTS_DIR" -type f -name "*.log" -o -name "*.txt" | head -n 1)"
    if [ -f "$LOG_FILE" ]; then
        echo "Found log file: $LOG_FILE"
        tail -f "$LOG_FILE"
    else
        echo "No log files found. Waiting for files to appear..."
        # Wait for any files to appear and monitor the directory
        while true; do
            LOG_FILE="$(find "$RESULTS_DIR" -type f -name "*.log" -o -name "*.txt" | head -n 1)"
            if [ -f "$LOG_FILE" ]; then
                echo "Found log file: $LOG_FILE"
                tail -f "$LOG_FILE"
                break
            fi
            sleep 2
            echo -n "."
        done
    fi
fi