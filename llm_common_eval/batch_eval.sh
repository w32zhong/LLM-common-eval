#!/bin/bash

detached_experiment() {
    CONDA_ENV=${1-default_conda_env}
    SESSION_ID=${2-default_session_id}
    shift 2
    CMD=${@-ls}
    echo "[new session=$SESSION_ID, conda_env=$CONDA_ENV] $CMD"
    tmux kill-session -t $SESSION_ID
    tmux new-session -c `pwd` -s $SESSION_ID -d
    tmux send-keys -t $SESSION_ID "conda activate $CONDA_DEFAULT_ENV" Enter
    tmux send-keys -t $SESSION_ID "$CMD" Enter
    tmux send-keys -t $SESSION_ID "exit" Enter
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
# detached_experiment base exp1 sleep 5
# detached_experiment base exp2 sleep 12
# waitfor_experiments exp1 exp2
