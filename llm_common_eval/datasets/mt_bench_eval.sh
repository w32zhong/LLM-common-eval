LOGDIR="$@"
source .env

function init() {
    #pip uninstall -y fschat
    cp mt_bench.patch mt_bench/fastchat/llm_judge
    pushd mt_bench/fastchat/llm_judge
    git apply mt_bench.patch 2> /dev/null
    mkdir -p ./data/mt_bench/model_answer
    cd data/mt_bench/model_answer && pwd && ls
    popd
}

function reset() {
    pushd mt_bench
    git checkout .
    popd
}

function convert() {
    outdir=./mt_bench/fastchat/llm_judge/data/mt_bench/model_answer
    for logdir in $LOGDIR; do
        if [[ "$logdir" =~ mt-bench ]]; then
            name=$(basename $logdir)
            python mt_bench_convert.py $logdir $name.jsonl \
                --out_dir $outdir
        fi
    done
    wc -l $outdir/*
}

function evaluate() {
    outdir=./mt_bench/fastchat/llm_judge/data/mt_bench/model_judgment
    wc -l $outdir/*

    set -e
    pushd mt_bench/fastchat/llm_judge
    python gen_judgment.py
    python show_result.py
    popd
}

init
convert
evaluate
reset
