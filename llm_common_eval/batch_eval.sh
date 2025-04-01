#!/bin/bash

detached_experiment() {
    SESSION_ID=${1-default_session_id}
    shift 1
    CMD="${@-ls}"
    CMD="$(eval echo $CMD)"
    tput bold setaf 3
    echo -n "[session=$SESSION_ID, conda_env=$CONDA_DEFAULT_ENV, devs=$CUDA_VISIBLE_DEVICES] "
    tput bold setaf 4
    echo $CMD
    tput sgr0
    tmux kill-session -t $SESSION_ID &> /dev/null
    tmux new-session -c `pwd` -s $SESSION_ID -d
    if [[ ! -z $CONDA_DEFAULT_ENV ]]; then
        tmux send-keys -t $SESSION_ID "conda activate $CONDA_DEFAULT_ENV" Enter
    fi
    tmux send-keys -t $SESSION_ID "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" Enter
    tmux send-keys -t $SESSION_ID "$CMD; exit" Enter
}

waitfor_experiments() {
    for exp in $@; do
        while tmux has-session -t $exp; do
            set -x
            tmux capture-pane -pt $exp -S -100 # show the last 100 lines
            set +x
            tput bold setaf 1
            echo "Waiting for tmux session $exp in $@ ..."
            tput sgr0
            sleep 5;
        done
    done
}

# Examples
# detached_experiment exp1 sleep 5
# detached_experiment exp2 sleep 12
# waitfor_experiments exp1 exp2

function block_until_set_available_devices() {
    assigner="$1"
    db_file="$2"
    output_file="$3"
    runid=$4
    budget=$5
    shift 5
    devs=""
    while [[ -z "$devs" ]]; do
        experiments="$(tmux list-sessions -F '#S' -f '#{m:exp*,#S}' | tr '\n' ' ')"
        echo "[experiments] $experiments"
        echo '[to inspect] tmux capture-pane -pt <experiment>'
        bash -c "$assigner refresh $experiments --db_file $db_file"
        bash -c "$assigner allocate $runid $budget --verbose True \
            --db_file $db_file --output_file $output_file"
        devs=$(cat $output_file)
        echo "[assigned available devices] $devs"
        sleep 5
    done
    export CUDA_VISIBLE_DEVICES=$devs
}
