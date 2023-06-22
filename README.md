# Alpaca Orca Open_LLaMa 3b
An Open_LLaMA-3B model trained on custom Alpaca dataset using Orca Research paper approaches.


## Dataset

We trained [OpenLLaMa-3B model](https://github.com/openlm-research/open_llama) on custom explain tuned Alpaca dataset (~52K) created using approaches from [Orca Research Paper](https://arxiv.org/abs/2306.02707). 

We leverage all of the 15 system instructions provided in [Orca Research Paper](https://arxiv.org/abs/2306.02707) to generate custom Alpaca dataset, in contrast to vanilla instruction tuning approaches used by original [Alpaca research paper](https://crfm.stanford.edu/2023/03/13/alpaca.html).

This helps student model aka [alpaca_orca_open_llama_3b](psmathur/alpaca_orca_open_llama_3b) to learn ***thought*** process from teacher model, which is ChatGPT (gpt-3.5-turbo-0301 version).

Please see below example usage how the **System** prompt is added before each *instruction*.

## Training

The training configurations are provided in the table below.

The training takes on 4x A600(50G) GPUs and lasts for around 20 Hours for cost of $66 using [Lambda Labs](https://lambdalabs.com)

We used DeepSpeed with Zero-3 approaches for parallel gpu training by writing our own fine tunning scripts plus leveraging some of the model training code provided by amazing [OpenAlpaca repo](https://github.com/yxuansu/OpenAlpaca)

Here are some of params used during training:

|||
|:-------------:|:-------------:|
|*batch_size*|16|
|*train_micro_batch_size_per_gpu*|2|
|*gradient_accumulation_steps*|2|
|*Learning rate*|2e-5|
|*Max length*|1024|
|*Epochs*|3|



## Example Usage

Below shows an example on how to use [alpaca_orca_open_llama_3b](psmathur/alpaca_orca_open_llama_3b)

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# change model_path between 3b,7b or 13b
model_path = 'psmathur/alpaca_orca_open_llama_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)


#generate text function
def generate_text(system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n#\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n#\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    print(f'[!] Response: {string}')

# same prompt as provided by Orca Research Paper
system = 'You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.'
instruction = 'Use the given data to calculate the median.'
input = '[5,2,3,4,1]'
generate_text(system, instruction, input)

```

### **P.S. I am #opentowork and #collaboration, if you can help, please reach out to me at psmathur.public@gmail.com**


### Next Goals:
1) Try more data, Dolly V2, WizardLM, & Others (we are open for suggestions)
2) Try bigger OpenLLaMA models 7B and 13B
3) Try better GPU for training, couldn't get 8xA100 (40GB), I guess they are in hot demand now.
4) Provide more options for Text generation UI. (may be https://github.com/oobabooga/text-generation-webui)
6) Provide 4bit GGML/GPTQ quantized model (may be [TheBloke](https://huggingface.co/TheBloke) can help here)


### Reference:
If you found [alpaca_orca_open_llama_3b](psmathur/alpaca_orca_open_llama_3b) useful in your research or applications, please kindly cite using the following BibTeX:

```
@misc{alpaca_orca_open_llama_3b,
  author = {Pankaj Mathur},
  title = {alpaca_orca_open_llama_3b: A custom explain tuned Alpaca Model Based On OpenLLaMA},
  year = {2023},
  publisher = {GitHub, HuggingFace},
  journal = {GitHub repository, HuggingFace repository},
  howpublished = {\url{https://github.com/pankajarm/alpaca_orca_open_llama_3b}, \url{https://https://huggingface.co/psmathur/alpaca_orca_open_llama_3b}},
}
```
```
@software{openlm2023openllama,
  author = {Xinyang Geng and Hao Liu},
  title = {OpenLLaMA: An Open Reproduction of LLaMA},
  month = May,
  year = 2023,
  url = {https://github.com/openlm-research/open_llama}
}
```
```
@misc{openalpaca,
  author = {Yixuan Su and Tian Lan and Deng Cai},
  title = {OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yxuansu/OpenAlpaca}},
}
```
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
