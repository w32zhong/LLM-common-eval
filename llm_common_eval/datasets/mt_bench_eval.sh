[[ ! $# -eq 1 ]] && echo 'bad arg.' && exit 1

LOGDIR=$1
name=foo

#pip uninstall -y fschat
source .env
cp mt_bench.patch mt_bench/fastchat/llm_judge
cd mt_bench/fastchat/llm_judge
git apply mt_bench.patch
mkdir -p ./data/mt_bench/model_answer
python mt_bench_convert.py $LOGDIR $name.jsonl
rm -rf data/mt_bench/model_judgment/
python gen_judgment.py --model-list $name
python show_result.py
