# bash
mkdir -p local_model

python -m model_train --model finance --lr 2e-5 --decay 0.02 --verbose
python -m model_train --model twitter --lr 2e-5 --decay 0.01 --verbose

mv model_result/fft-model_finance-lr_2e-05-decay_0.02 local_model/fine-tuned_finance
mv model_result/fft-model_twitter-lr_2e-05-decay_0.01 local_model/fine-tuned_twitter