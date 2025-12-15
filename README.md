# Sentiment-stock
This repository is about fine-tuning financial-specific and tweet-specific pretrained models on financial tweets to compare against another model pretrained solely on financial tweets.

## Financial tweets dataset
Download financial tweets dataset from this [dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) [[4]](#4)

```bash
python -m dataset_download --verbose
```

## Baseline Models
With the test set from the dataset, we can test model performance with

```bash
python -m model_test --model [model] --verbose
```

where `[model]` is

* [twitter](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) [[2]](#2)
* [finance](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis) [[3]](#3)
* [fintwit](https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment) [[1]](#1)

## Fine-tuned Models
Due to file size limitations, fine-tuned models are not uploaded to the repository. To reproduce the results, please run:

```bash
bash local_model.sh
```

Models in the `local_model` directory correspond to checkpoints that achieve the best validation accuracy after training with different learning rates and weight decay (L2 Regularization)

```bash
python -m model_test --local_model [model] --verbose
```

where `[model]` is

* local_model/fine-tuned_finance
* local_model/fine-tuned_twitter

## Results
Evaluation is performed using the test split of the dataset, with accuracy as the primary metric.

| Model                   | Accuracy | F1-score |
|-------------------------|----------|----------|
| Twitter-based [[2]](#2) | 0.709    | 0.617    |
| Finance-based [[3]](#3) | 0.744    | 0.652    |
| FinTwit-based [[1]](#1) | 0.864    | 0.831    |
| Fine-tuned Twitter      | 0.899    | 0.867    |
| Fine-tuned Finance      | 0.849    | 0.787    |

Fine-tuning models on financial tweets significantly improves performance compared to their pretrained versions, with the fine-tuned Twitter-based model outperforming the model pretrained on both finance and Twitter data.

## References
<a id="1">[1]</a> Akkerman, S. & Koornstra, T. (2023). FinTwitBERT-sentiment: A Sentiment Classifier for Financial Tweets. *Hugging Face.* https://huggingface.co/StephanAkkerman/FinTwitBERT-sentiment

<a id="2">[2]</a> Camacho-collados, J., Rezaee, K., Riahi, T., Ushio, A., Loureiro, D., Antypas, D., Boisson, J., Espinosa Anke, L., Liu, F., Martínez Cámara, E., Medina, G., Buhrmann, T., Neves, L. & Barbieri, F. (2022). TWEETNLP: Cutting-edge natural language processing for social media. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.* https://doi.org/10.18653/v1/2022.emnlp-demos.5

<a id="3">[3]</a> Hazourli, Ahmed. (2022). *FinancialBERT - A Pretrained Language Model for Financial Text Mining*. 10.13140/RG.2.2.34032.12803.

<a id="4">[4]</a> Hugging Face. (2024). *zeroshot/twitter-financial-news-sentiment* (Version 1.0.0) [Dataset]. https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
