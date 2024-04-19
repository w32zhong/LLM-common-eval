### About
Some well-known datasets are unavailable on the HuggingFace hub, to support the evaluation on more datasets, some new HuggingFace datasets are created via scripts in this directory.

Following the format of HuggingFace dataset will make our model evaluation pipeline more consistent. Here is a list of resources we refer to create commonly used datasets that have not been created on the HuggingFace yet:

* For long context length: https://huggingface.co/amazon/MistralLite

### MT-Bench
To mock the output data and judge:
```sh
cp mt_bench_mock_* mt_bench/fastchat/llm_judge
cd mt_bench
pip install -e ".[model_worker,llm_judge]"
cd fastchat/llm_judge
git apply mt_bench_mock_judge.patch
mkdir -p ./data/mt_bench/model_answer/
python mt_bench_mock_answer.py
rm -rf data/mt_bench/model_judgment/
python gen_judgment.py --model-list foo
python show_result.py
```
