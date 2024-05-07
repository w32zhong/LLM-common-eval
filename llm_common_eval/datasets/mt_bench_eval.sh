[[ ! $# -eq 1 ]] && echo 'bad arg.' && exit 1

LOGDIR=$1
name=foo

cp mt_bench_* mt_bench/fastchat/llm_judge
cd mt_bench/fastchat/llm_judge
git apply mt_bench_mock_judge.patch
mkdir -p ./data/mt_bench/model_answer
python mt_bench_convert.py $LOGDIR $name.jsonl
rm -rf data/mt_bench/model_judgment/
python gen_judgment.py --model-list $name
python show_result.py
