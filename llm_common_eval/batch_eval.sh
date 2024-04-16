#!/bin/bash

detached_experiment() {
    SESSION_ID=${1-default_session_id}
    shift 1
    CMD=${@-ls}
    echo "[new session=$SESSION_ID, conda_env=$CONDA_DEFAULT_ENV] $CMD"
    tmux kill-session -t $SESSION_ID &> /dev/null
    tmux new-session -c `pwd` -s $SESSION_ID -d
    tmux send-keys -t $SESSION_ID "conda activate $CONDA_DEFAULT_ENV" Enter
    tmux send-keys -t $SESSION_ID "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" Enter
    tmux send-keys -t $SESSION_ID "$CMD" Enter
}

waitfor_experiments() {
    for exp in $@; do
        while tmux has-session -t $exp; do
            set -x
            tmux capture-pane -pt $exp -S -100 # show the last 100 lines
            set +x
            tput bold setaf 1
            echo "Waiting for tmux session: $exp ..."
            tput sgr0
            sleep 5;
        done
    done
}

# Examples
# detached_experiment exp1 sleep 5
# detached_experiment exp2 sleep 12
# waitfor_experiments exp1 exp2
