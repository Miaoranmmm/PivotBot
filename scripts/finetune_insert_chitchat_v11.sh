setting="insert_chitchat"
task=${1}
start_idx=${2}
fewshot=${3}
data_file=${4}
seed=${5}
if [ $task == "chitchat" ];
then
	model='chatbot'
else
	model="${task}"
fi
store_path="newdata"
if [[ $data_file =~ "_no_kw" ]]; then
	store_path="newdata-no-kw"
fi
python finetune_model_v11_new_criteria.py \
	--seed=${seed} \
	--num_train_epochs=15 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=8 \
	--dataset_name="data/${data_file}.py" \
	--model_name_or_path="microsoft/GODEL-v1_1-base-seq2seq" \
	--output_dir="models_new_criteria/${setting}/${model}/unified/${store_path}_${seed}/GODEL-V1.1/fewshot-${fewshot}/start-${start_idx}"  \
	--dataset_config_name="${setting}-${task}-${start_idx}-${fewshot}"