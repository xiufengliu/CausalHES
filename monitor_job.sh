#!/bin/bash
# Monitor script for enhanced baseline experiments

echo "Enhanced Baseline Experiments - Job Monitor"
echo "==========================================="

JOB_ID=$(bjobs -J enhanced_baseline_experiments -o "JOBID" -noheader 2>/dev/null | head -1)

if [ -z "$JOB_ID" ]; then
    echo "No job found with name 'enhanced_baseline_experiments'"
    echo "Checking for any recent jobs..."
    bjobs -u xiuli
    exit 1
fi

echo "Monitoring job ID: $JOB_ID"
echo "Job name: enhanced_baseline_experiments"
echo ""

while true; do
    STATUS=$(bjobs $JOB_ID -o "STAT" -noheader 2>/dev/null)
    
    if [ -z "$STATUS" ]; then
        echo "Job $JOB_ID has completed!"
        echo ""
        echo "Checking for output files..."
        if [ -f "enhanced_baseline_${JOB_ID}.out" ]; then
            echo "=== OUTPUT FILE FOUND ==="
            echo "File: enhanced_baseline_${JOB_ID}.out"
            echo "Size: $(wc -l enhanced_baseline_${JOB_ID}.out | cut -d' ' -f1) lines"
            echo ""
            echo "=== LAST 20 LINES OF OUTPUT ==="
            tail -20 "enhanced_baseline_${JOB_ID}.out"
        else
            echo "Output file not found yet"
        fi
        
        if [ -f "enhanced_baseline_${JOB_ID}.err" ]; then
            echo ""
            echo "=== ERROR FILE FOUND ==="
            echo "File: enhanced_baseline_${JOB_ID}.err"
            echo "Size: $(wc -l enhanced_baseline_${JOB_ID}.err | cut -d' ' -f1) lines"
            if [ -s "enhanced_baseline_${JOB_ID}.err" ]; then
                echo ""
                echo "=== ERROR CONTENT ==="
                cat "enhanced_baseline_${JOB_ID}.err"
            else
                echo "Error file is empty (good!)"
            fi
        fi
        
        echo ""
        echo "Checking for results directory..."
        if [ -d "experiments/results/enhanced_baselines" ]; then
            echo "Results directory found:"
            ls -la experiments/results/enhanced_baselines/
        fi
        
        break
    fi
    
    echo "$(date): Job $JOB_ID status: $STATUS"
    
    # If job is running, try to show some output
    if [ "$STATUS" = "RUN" ] && [ -f "enhanced_baseline_${JOB_ID}.out" ]; then
        echo "  Output lines so far: $(wc -l enhanced_baseline_${JOB_ID}.out | cut -d' ' -f1)"
        echo "  Latest output:"
        tail -3 "enhanced_baseline_${JOB_ID}.out" | sed 's/^/    /'
    fi
    
    sleep 30
done

echo ""
echo "Monitoring complete!"
