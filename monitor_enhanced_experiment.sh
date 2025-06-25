#!/bin/bash

# Monitor the enhanced baseline experiment job
JOB_ID=25327222

echo "Enhanced Baseline Experiment Monitoring"
echo "========================================"
echo "Job ID: $JOB_ID"
echo "Time: $(date)"
echo ""

# Check job status
echo "Job Status:"
bjobs $JOB_ID 2>/dev/null || echo "Job not found (may have completed)"
echo ""

# Check output file sizes
echo "Output Files:"
ls -lh enhanced_baseline_${JOB_ID}.out enhanced_baseline_${JOB_ID}.err 2>/dev/null
echo ""

# Show last 10 lines of output
echo "Latest Output (last 10 lines):"
echo "------------------------------"
tail -n 10 enhanced_baseline_${JOB_ID}.out 2>/dev/null || echo "No output file found"
echo ""

# Show errors if any
if [ -s enhanced_baseline_${JOB_ID}.err ]; then
    echo "Errors:"
    echo "-------"
    tail -n 10 enhanced_baseline_${JOB_ID}.err
else
    echo "No errors detected"
fi

echo ""
echo "========================================"
