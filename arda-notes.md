### Day 1
We will use habanoz/tr-news-1.8m news dataset for our first models. I will use 1/5 of whole dataset for tokenizer training and probably 1/4 for the model pre-training. These number are selected because of low compute resource. I will be training these models on Google Colab, so our most powerfull hardware is A100 GPU with 80GB vRAM. 

First things first I am creating a tokenizer to train. As I mentioned above I will be using 1/5 of the dataset I have. I will upload tokenizer.json of this tokenizer to HuggingFace in order to access easily in future. I have found out that uploading tokenizer.json standalone may cause problems when trying to load it as PretrainedTokenizerFast. So I will use "PretrainedTokenizerFast.push_to_hub()". In order to upload it in right format with files like config.json.

After 10 minutes, I have noticed that I didn't specified the vocab size of tokenizer. So I interrupted the training and specifiying the vocab size as 32768.

The training has done, it took 33 minutes. I pushed it to HuggingFace as "aliarda/turkish-news-32k-tokenizer". Also training codes for tokenizer are at turkishNews32Tokenizer.ipynb 

Next step, I will tokenize our dataset with turkishNews32Tokenizer and upload that to HuggingFace too.