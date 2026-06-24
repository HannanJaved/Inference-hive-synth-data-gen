#!/bin/bash -l
#SBATCH --job-name=datamix-synth-1b-topup
#SBATCH --partition=alpha
##SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --signal=B:SIGUSR1@120
#SBATCH --output=/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/logs/%a-%A-%N.log
#SBATCH --error=/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/logs/%a-%A-%N.log


# Print job information
echo "=== SLURM Job Information ==="
echo "SLURM_JOB_NAME: ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION}"
echo "SLURM_JOB_ACCOUNT: ${SLURM_JOB_ACCOUNT}"
echo "============================="

# Check if this shard is already completed
CURRENT_SHARD=${SLURM_ARRAY_TASK_ID}
COMPLETED_SHARDS_FILE="/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/progress/shards_completed.log"
FAILED_SHARDS_FILE="/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/progress/shards_failed.log"


set -e

log() {
   local level="$1"
   local message="$2"
   echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message"
}

log_failed_shard() {
    local reason="$1"
    local additional_info="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local job_info="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID}"
    
    # Create failed shards log entry: shard_number timestamp reason job_info hostname additional_info
    echo "${CURRENT_SHARD} ${timestamp} ${reason} ${job_info} ${additional_info}" >> "${FAILED_SHARDS_FILE}"
    log "ERROR" "Marked shard ${CURRENT_SHARD} as failed: ${reason} ${additional_info}"
}

# Check if this shard is already completed
if [ -f "$COMPLETED_SHARDS_FILE" ]; then
    # Use grep with word boundaries to match exact shard number
    if grep -q "^${CURRENT_SHARD}$" "$COMPLETED_SHARDS_FILE"; then
        log "INFO" "Shard ${CURRENT_SHARD} is already completed. Exiting."
        exit 0
    else
        log "INFO" "Shard ${CURRENT_SHARD} not found in completed shards. Proceeding with inference."
    fi
else
    log "INFO" "No completed shards log found. Proceeding with inference."
fi

cleanup() {
    local signal=$1
    log "INFO" "Received signal $signal, initiating cleanup..."
    if [ "$signal" = "SIGUSR1" ]; then
        log "WARN" "Job is about to hit time limit, shutting down gracefully..."
    elif [ "$signal" = "SIGTERM" ]; then
        log "WARN" "Job received cancellation signal, shutting down gracefully..."
    else
        log "WARN" "Job received signal $signal, shutting down gracefully..."
    fi
    log "INFO" "Initiating graceful shutdown of processes..."
    
    # Send SIGINT to inference process group
    if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
        log "INFO" "Sending SIGINT to inference process group (PID: $INFERENCE_PID)"
        # Send signal to the entire process group using negative PID
        kill -INT -$INFERENCE_PID 2>/dev/null || kill -INT $INFERENCE_PID
    fi
    
    # give the inference script some time before shutting down the inference server
    sleep 10 

    # Send two SIGINTs to inference server srun process group (srun expects two to abort)
    if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
        log "INFO" "Sending SIGINT x2 to inference server srun process group (PID: $INFERENCE_SERVER_PID)"
        # Send signal to the entire process group using negative PID
        kill -INT -$INFERENCE_SERVER_PID 2>/dev/null || kill -INT $INFERENCE_SERVER_PID
        # Short delay to ensure the second INT is within 1s window expected by srun
        sleep 0.1
        if kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            kill -INT -$INFERENCE_SERVER_PID 2>/dev/null || kill -INT $INFERENCE_SERVER_PID
        fi
    fi

    log "INFO" "Waiting for processes to finish gracefully..."

    # Wait for processes to finish with timeout
    local wait_timeout=60
    local wait_count=0
    
    while [ $wait_count -lt $wait_timeout ]; do
        local processes_running=0
        
        # Check if processes are still running
        if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        if [ ! -z "$HEALTHCHECK_PID" ] && kill -0 $HEALTHCHECK_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
            processes_running=$((processes_running + 1))
        fi
        
        if [ $processes_running -eq 0 ]; then
            log "INFO" "All processes have finished gracefully"
            break
        fi
        
        log "INFO" "Waiting for $processes_running processes to finish... ($wait_count/$wait_timeout)"
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    # Force kill any remaining processes
    if [ $wait_count -eq $wait_timeout ]; then
        log "WARN" "Timeout reached, force killing remaining processes"
        if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            # Force kill the entire process group
            kill -KILL -$INFERENCE_SERVER_PID 2>/dev/null || kill -KILL $INFERENCE_SERVER_PID 2>/dev/null
        fi
        if [ ! -z "$INFERENCE_PID" ] && kill -0 $INFERENCE_PID 2>/dev/null; then
            # Force kill the entire process group
            kill -KILL -$INFERENCE_PID 2>/dev/null || kill -KILL $INFERENCE_PID 2>/dev/null
        fi
    fi
    
    # Only resubmit if this was a time limit signal, not a manual cancellation
    # Resubmit instead of requeue since requeue is disabled on many clusters.
    sbatch_command=(sbatch --array="${SLURM_ARRAY_TASK_ID}" --job-name="datamix-synth-1b-topup-$(printf '%06d' "${SLURM_ARRAY_TASK_ID}")" "/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/ih_job.slurm")
    if [ "$signal" = "SIGUSR1" ]; then
        log "INFO" "Resubmitting task ${SLURM_ARRAY_TASK_ID} automatically due to time limit..."
        log "INFO" "Running command: ${sbatch_command[*]}"
        if sbatch_output="$("${sbatch_command[@]}" 2>&1)"; then
            new_job_id=$(echo "$sbatch_output" | grep -o '[0-9]*')
            log "INFO" "Task ${SLURM_ARRAY_TASK_ID} resubmitted successfully as job ${new_job_id}. Progress will resume from checkpoint."
        else
            log "ERROR" "Failed to resubmit task ${SLURM_ARRAY_TASK_ID}. Error: $sbatch_output"
            log "ERROR" "You may need to resubmit manually: ${sbatch_command[*]}"
            # Mark as failed since resubmission failed
            log_failed_shard "resubmission_failed" "Failed to resubmit after time limit: $sbatch_output"
        fi
    else
        # Mark as failed for any signal other than SIGUSR1 (time limit)
        if [ "$signal" = "SIGTERM" ]; then
            log_failed_shard "manual_cancellation" "Job was manually cancelled"
        else
            log_failed_shard "unexpected_signal" "Job received signal: $signal"
        fi
        log "INFO" "Task ${SLURM_ARRAY_TASK_ID} was manually cancelled - not resubmitting automatically."
        log "INFO" "To restart this task later, run: ${sbatch_command[*]}"
    fi
    
    exit 0
}

trap 'cleanup SIGUSR1' SIGUSR1 # send before timelimit is hit
trap 'cleanup SIGTERM' SIGTERM # send by scancel

# Setup env

# Activate environment
log "INFO" "Activating environment with custom command"
module load CUDA
source /data/horse/ws/hama901h-whittle/cache.sh
source /data/horse/ws/hama901h-whittle/inference-hive/.venv/bin/activate

log "INFO" "python path: $(which python)"

# Add env exports
export HF_HUB_OFFLINE="1"
export VLLM_DISABLE_COMPILE_CACHE="1"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


# Pick a random port in valid range to avoid collisions when multiple shards run on same node
# Range 30000-60000 avoids privileged ports and stays well under the 65535 limit
IH_PORT=$((30000 + RANDOM % 30000))
export IH_PORT
log "INFO" "Dynamic port allocation: IH_PORT=$IH_PORT"

# Set API_BASE_URL with dynamic port substitution
# Replace ${IH_PORT} in the URL template with the actual port value
_API_BASE_URL_TEMPLATE="http://localhost:${IH_PORT}/v1"
export API_BASE_URL="${_API_BASE_URL_TEMPLATE//\$\{IH_PORT\}/$IH_PORT}"
log "INFO" "API_BASE_URL=$API_BASE_URL"

MASTER_NODE=$(scontrol show hostname ${SLURM_JOB_NODELIST} | head -n1)
export MASTER_NODE

# Validate dataset before starting inference server
log "INFO" "Validating dataset format for shard ${CURRENT_SHARD}"
python validate_data.py --config /data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/ih_config.yaml --shard ${CURRENT_SHARD} --num-shards 16
VALIDATION_EXIT_CODE=$?

if [ $VALIDATION_EXIT_CODE -ne 0 ]; then
    log "ERROR" "Dataset validation failed for shard ${CURRENT_SHARD}. Exiting."
    log_failed_shard "data_validation_failed" "Dataset validation failed before starting inference server"
    exit 1
fi

log "INFO" "Dataset validation passed for shard ${CURRENT_SHARD}"

# Start inference server
log "INFO" "Starting inference server on ${SLURM_JOB_NUM_NODES} nodes"
INFERENCE_SERVER_LOG="/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/logs/${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}-%N-inference-server.log"
INFERENCE_SERVER_COMMAND="python -m vllm.entrypoints.openai.api_server --host=localhost --port=$IH_PORT --model="/data/horse/ws/hama901h-whittle/.cache/huggingface/hub/models--openeurollm--datamix-9b-80-20/snapshots/66c3229e708857a0bc54070131d4b1762d4f5279" --tensor-parallel-size=1 --data-parallel-size=1 --max-model-len=2048 --trust-remote-code --gpu-memory-utilization=0.9 --max-log-len=10"
log "INFO" "Inference server command: ${INFERENCE_SERVER_COMMAND}"
setsid srun --output="$INFERENCE_SERVER_LOG" --error="$INFERENCE_SERVER_LOG" \
    bash -c "${INFERENCE_SERVER_COMMAND}" &
INFERENCE_SERVER_PID=$!

# Give the inference server srun command time to actually start
sleep 5

# Check if the srun command is still running (not the inference server itself, but the srun process)
if ! kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
    # The srun command itself has exited, check its exit code
    wait $INFERENCE_SERVER_PID
    INFERENCE_SERVER_EXIT_CODE=$?
    if [ $INFERENCE_SERVER_EXIT_CODE -ne 0 ]; then
        log "ERROR" "Inference server srun command failed with exit code $INFERENCE_SERVER_EXIT_CODE. Check the inference server logs for details."
        log "ERROR" "This usually means the inference server failed to start. Exiting to prevent resource waste."
        log_failed_shard "inference_server_startup_failed" "srun command failed with exit code $INFERENCE_SERVER_EXIT_CODE"
        exit 1
    fi
fi

# Wait for inference server to become healthy
wait_for_api_server() {
    local max_attempts=$((20 * 60 / 20))  # Calculate attempts based on total time and interval
    log "INFO" "Maximum health check wait time: 20 minutes"
    local attempt=0
    local health_url="${API_BASE_URL%/v1}/health"
    
    while [ $attempt -lt $max_attempts ]; do
        log "INFO" "Health check attempt $((attempt + 1))/${max_attempts} for ${health_url}"
        
        # Check if the inference server srun process is still running
        if ! kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
            log "ERROR" "Inference server srun process (PID: $INFERENCE_SERVER_PID) has terminated during health checks"
            # Wait for the process to get its exit code
            wait $INFERENCE_SERVER_PID 2>/dev/null
            INFERENCE_SERVER_EXIT_CODE=$?
            log "ERROR" "Inference server srun command exited with code $INFERENCE_SERVER_EXIT_CODE"
            log_failed_shard "inference_server_terminated_during_healthcheck" "srun process died during health checks with exit code $INFERENCE_SERVER_EXIT_CODE"
            return 1
        fi
        
        if curl -s --connect-timeout 5 --max-time 10 "${health_url}" >/dev/null 2>&1; then
            log "INFO" "Inference server is healthy and ready!"
            return 0
        fi
        
        log "INFO" "Inference server not ready, waiting 20 seconds..."
        sleep 20
        attempt=$((attempt + 1))
    done
    
    log "ERROR" "Error: inference server did not become healthy within timeout"
    return 1
}
log "INFO" "Waiting for inference server to become healthy"
wait_for_api_server
HEALTHCHECK_EXIT_CODE=$?

# If health check failed, exit immediately
if [ $HEALTHCHECK_EXIT_CODE -ne 0 ]; then
    log "ERROR" "Health check failed. Exiting job to prevent resource waste."
    log_failed_shard "health_check_failed" "Inference server did not become healthy within timeout"
    exit 1
fi

# Run inference
log "INFO" "Running inference"
INFERENCE_PROGRESS_LOG="/data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/progress/${SLURM_ARRAY_TASK_ID}-progress.jsonl"
# Start inference in a new process group so we can kill the entire group
setsid python run_inference.py --config /data/horse/ws/hama901h-whittle/inference-hive/examples/datamix_1b_synthetic/runs/datamix_1b_topup/ih_config.yaml --num-shards 16 --shard ${CURRENT_SHARD} --log-file $INFERENCE_PROGRESS_LOG &
INFERENCE_PID=$!

# GPU idle monitor - background watchdog that terminates the job if the GPU
# idles for too long (e.g. inference server crashed but client keeps retrying).
# Waits for a warm-up period, then polls nvidia-smi every 60s.
(
    GPU_IDLE_WARMUP=300
    GPU_IDLE_THRESHOLD=5
    GPU_IDLE_MAX_CHECKS=5
    GPU_IDLE_INTERVAL=60

    sleep $GPU_IDLE_WARMUP
    consecutive_idle=0
    while kill -0 $INFERENCE_PID 2>/dev/null; do
        gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$gpu_util" ] && [ "$gpu_util" -lt $GPU_IDLE_THRESHOLD ] 2>/dev/null; then
            consecutive_idle=$((consecutive_idle + 1))
            log "WARN" "GPU utilization low (${gpu_util}%), idle check ${consecutive_idle}/${GPU_IDLE_MAX_CHECKS}"
            if [ $consecutive_idle -ge $GPU_IDLE_MAX_CHECKS ]; then
                log "ERROR" "GPU idle for $((GPU_IDLE_MAX_CHECKS * GPU_IDLE_INTERVAL))s+ — inference server likely dead. Terminating to save resources."
                log_failed_shard "gpu_idle_timeout" "GPU utilization below ${GPU_IDLE_THRESHOLD}% for $((GPU_IDLE_MAX_CHECKS * GPU_IDLE_INTERVAL))s"
                kill -TERM -$INFERENCE_PID 2>/dev/null || kill -TERM $INFERENCE_PID 2>/dev/null
                exit 0
            fi
        else
            consecutive_idle=0
        fi
        sleep $GPU_IDLE_INTERVAL
    done
) &
HEALTHCHECK_PID=$!

wait $INFERENCE_PID
INFERENCE_EXIT_CODE=$?

# Stop the GPU idle monitor (use || true to avoid set -e exit when process already gone)
if [ -n "$HEALTHCHECK_PID" ]; then
    kill $HEALTHCHECK_PID 2>/dev/null || true
    wait $HEALTHCHECK_PID 2>/dev/null || true
fi

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    log "INFO" "Inference completed successfully, recording shard completion"
    echo "${CURRENT_SHARD}" >> "$COMPLETED_SHARDS_FILE"
    log "INFO" "Done"
else
    log "ERROR" "Inference failed with exit code $INFERENCE_EXIT_CODE"
    log_failed_shard "inference_script_failed" "run_inference.py exited with code $INFERENCE_EXIT_CODE"
fi

# Cleanup: terminate the inference server regardless of how inference ended
# This is critical to avoid wasting GPU hours when inference fails
log "INFO" "Shutting down inference server..."
if [ ! -z "$INFERENCE_SERVER_PID" ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
    log "INFO" "Sending SIGINT to inference server (PID: $INFERENCE_SERVER_PID)"
    kill -INT -$INFERENCE_SERVER_PID 2>/dev/null || kill -INT $INFERENCE_SERVER_PID 2>/dev/null
    sleep 0.1
    if kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
        kill -INT -$INFERENCE_SERVER_PID 2>/dev/null || kill -INT $INFERENCE_SERVER_PID 2>/dev/null
    fi
    
    # Wait up to 30 seconds for graceful shutdown
    wait_count=0
    while [ $wait_count -lt 30 ] && kill -0 $INFERENCE_SERVER_PID 2>/dev/null; do
        sleep 1
        wait_count=$((wait_count + 1))
    done
    
    # Force kill if still running
    if kill -0 $INFERENCE_SERVER_PID 2>/dev/null; then
        log "WARN" "Inference server did not terminate gracefully, force killing..."
        kill -KILL -$INFERENCE_SERVER_PID 2>/dev/null || kill -KILL $INFERENCE_SERVER_PID 2>/dev/null
    fi
fi
log "INFO" "Inference server shutdown complete"

# Exit with the inference exit code to properly signal success/failure to Slurm
exit $INFERENCE_EXIT_CODE
