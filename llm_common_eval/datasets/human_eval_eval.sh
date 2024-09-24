LOGDIR="$@"

function init() {
    cp human_eval.patch human_eval/
    pushd human_eval
    git apply human_eval.patch
    popd
}

function reset() {
    cp human_eval.patch human_eval/
    pushd human_eval
    rm -rf data
    git checkout .
    popd
}

function convert() {
    outdir='./human_eval/data'
    for logdir in $LOGDIR; do
        if [[ "$logdir" =~ human-eval ]]; then
            name=$(basename $logdir)
            python human_eval_convert.py extract_problems $logdir problems-$name.jsonl \
                --out_dir $outdir
            python human_eval_convert.py convert $logdir $name.jsonl \
                --out_dir $outdir
        fi
    done
    wc -l $outdir/*
}

function evaluate() {
    pushd human_eval
    for file in ./data/experiment-*.jsonl; do
        set -x
        python -m human_eval.evaluate_functional_correctness $file
        set +x
    done
    popd
}

init
convert
evaluate
reset
cat ./human_eval/results.txt
