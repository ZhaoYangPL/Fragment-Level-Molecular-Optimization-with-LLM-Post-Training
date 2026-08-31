#!/usr/bin/env fish

set -l script_dir (dirname (status --current-filename))
cd $script_dir

set -l resume_flag
if test (count $argv) -gt 1
    echo "Usage: ./run_v2_phase1_4000.fish [--resume]" >&2
    exit 2
end

if test (count $argv) -eq 1
    if test "$argv[1]" = "--resume"
        set resume_flag --resume
    else
        echo "Usage: ./run_v2_phase1_4000.fish [--resume]" >&2
        exit 2
    end
end

python -u dataset/dataset_constuction_v2.py \
    --num-train 4000 \
    --num-val 200 \
    --output-train datasets_v2/phase1_4000_intent/fourstep_qed_logp_mw_sft_train_v2_intent_4000.jsonl \
    --output-val datasets_v2/phase1_4000_intent/fourstep_qed_logp_mw_sft_val_v2_intent_200.jsonl \
    --checkpoint-every 500 \
    --split-offset 0 \
    --manifest-path datasets_v2/phase1_4000_intent/manifest_v2_intent_4000_200_offset0.json \
    --plan-init-timeout 240 \
    --plan-result-timeout 120 \
    $resume_flag
