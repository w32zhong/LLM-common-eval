source ./llm_common_eval/batch_eval.sh

# Dataset validation size (ordered):
# cb        56
# copa      100
# wsc       104
# rte       277
# wic       638
# boolq     3.27K
# multirc   4.85K
# record    10K

detached_experiment lg exp1 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py cb --devices 0,1
waitfor_experiments exp1
exit

detached_experiment lg exp2 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py copa --devices 0,1
detached_experiment lg exp3 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py wsc --devices 2,3
waitfor_experiments exp2 exp3

detached_experiment lg exp4 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py rte --devices 0,1
detached_experiment lg exp5 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py wic --devices 2,3
waitfor_experiments exp4 exp5

detached_experiment lg exp6 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py boolq --devices 0,1
detached_experiment lg exp7 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py multirc --devices 2,3
waitfor_experiments exp6 exp7

detached_experiment lg exp8 \
    python ./eval_phi2/phi2_on_SuperGLUE_cuda_allsubset.py record --devices 0,1
