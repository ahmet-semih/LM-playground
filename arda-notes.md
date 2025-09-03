### Day 1
We will use habanoz/tr-news-1.8m news dataset for our first models. I will use 1/5 of whole dataset for tokenizer training and probably 1/4 for the model pre-training. These number are selected because of low compute resource. I will be training these models on Google Colab, so our most powerfull hardware is A100 GPU with 80GB vRAM. 

First things first I am creating a tokenizer to train. As I mentioned above I will be using 1/5 of the dataset I have. I will upload tokenizer.json of this tokenizer to HuggingFace in order to access easily in future. I have found out that uploading tokenizer.json standalone may cause problems when trying to load it as PretrainedTokenizerFast. So I will use "PretrainedTokenizerFast.push_to_hub()". In order to upload it in right format with files like config.json.

After 10 minutes, I have noticed that I didn't specified the vocab size of tokenizer. So I interrupted the training and specifiying the vocab size as 32768.

The training has done, it took 33 minutes. I pushed it to HuggingFace as "aliarda/turkish-news-32k-tokenizer". Also training codes for tokenizer are at turkishNews32Tokenizer.ipynb 

Next step, I will tokenize our dataset with turkishNews32Tokenizer and upload that to HuggingFace too. I tokenized the whole dataset and it took 29 minutes using fast tokenizer. I just realized that I made a mistake again, I tokenized whole dataset but did not saved the changes, so it's all lost. I have to tokenize again. 

I found out what was the issue. When I run a for loop on dataset it takes each rows copy and process on that, not the dataset itself, so I used datasets.map() function to re-tokenize entire dataset. 

Tokenizing the dataset is done and I shared it on HF as "aliarda/turkish-news-1.8M-tokenized".

Now, I have to initialize a llama model with ~50M parameters. This will be our checkpoint 0 and I will upload it to HuggingFace. 

It is now on HF as "aliarda/llama-50M-randParams", with 51,521,792 parameters. 

### Day 2
I will start training of models today. I will write a dataloader and training code, I am going to also use twilio to receive WhatsApp messages when each epoch is done. 

I have shuffled and took 1/4 of tokenized training data for train, then I split the data to 10 chunks. I will train model on each chunk 2 epoch with in total of 20 epochs. 

I started training code but as soon as I run the cell I got an error saying the shape is not compatible with input 512. I did not understood what it said and couldn't figure out after ~10minutes. Then I asked to ChatGPT and it told that (emb_dims / n_heads) must be at least 2. Mine was 1 (256/256) so I reconfigured the model with 128 attention heads.  

After I solve that problem, I got stuck with CUDA out of memory. I have tried decreasing batch size 8x or halving the context length but couldn't solve that. I guess I am going to continue tomorrow. 
