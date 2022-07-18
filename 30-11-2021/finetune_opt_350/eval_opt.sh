OUTPUT_DIR=model/finetuned_opt-350-01/eval
MODEL_PATH=model/pre_trained_model_opt-350M/
#MODEL_PATH=model/finetuned_opt-350-01/
TEST_FILE=data/test.txt
python src/run_clm.py \
--output_dir=$OUTPUT_DIR \
--model_name_or_path=$MODEL_PATH \
--do_eval \
--validation_file=$TEST_FILE \
--report_to wandb \
--run_name finetuned_opt-350M_evel
#--max_eval_samples 3 \