MODEL_PATH=model/pre_trained_model_opt-350M/
#MODEL_PATH=model/finetuned_opt-350-01/

K=50
python src/run_generation.py \
--model_type opt \
--model_name_or_path $MODEL_PATH \
--length 100 \
--prefix "[BOS]" \
--sep_token "[SEP]" \
--stop_token "[EOS]" \
--first_sentence "The weather is good today." \
--last_sentence "It really deserved to go out." \
--k $K \
--num_return_sequences 5