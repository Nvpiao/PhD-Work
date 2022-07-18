OUTPUT_DIR=model/finetuned_opt-350-01/
MODEL_PATH=model/pre_trained_model_opt-350M/
TRAIN_FILE=data/train.txt
VALID_FILE=data/valid.txt
python src/run_clm.py \
--output_dir=$OUTPUT_DIR \
--model_name_or_path=$MODEL_PATH \
--do_train \
--train_file=$TRAIN_FILE \
--do_eval \
--validation_file=$VALID_FILE \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--evaluation_strategy steps \
--logging_steps 8 \
--learning_rate=5e-5 \
--num_train_epochs=5 \
--report_to wandb \
--run_name finetuned_opt-350M