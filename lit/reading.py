from dotenv import load_dotenv
load_dotenv()
import json
import os
import fire

import numpy as np
import torch
from transformers import PreTrainedModel

from lit.utils.dataset_utils import lqa_tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
from lit.utils.activation_utils import latent_qa
from lit.utils.infra_utils import (
    update_config,
    get_model,
    get_tokenizer,
    get_modules,
    load_sae,
    clean_text,
)
from lit.utils.my_dataset_utils import *
import sys

def messages_to_string(messages):
    formatted_messages = []
    for message in messages:
        role = message.get("role", "unknown").capitalize()  # e.g., "user" -> "User"
        content = message.get("content", "")
        formatted_messages.append(f"{role}: {content}")
    
    full_prompt = "\n".join(formatted_messages)
    
    return full_prompt

def interpret(
    target_model,
    decoder_model,
    tokenizer,
    dialogs,
    questions,
    args,
    generate=True,
    sae=None,
):

    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    module_read, module_write = get_modules(target_model, decoder_model, **vars(args))

    save_sae_dist = getattr(args, "save_sae_distribution", False) and sae is not None
    if save_sae_dist:
        os.makedirs("controls", exist_ok=True)

    out = []
    for batch_indices in batch_index_generator(len(questions), args.batch_size):
        print(batch_indices)
        probe_data = []
        for idx in batch_indices:
            rp, qa = dialogs[idx], questions[idx]
            question = [{"role": "user", "content": qa[0]}]
            # try:
            print(rp)
            read_prompt = tokenizer.apply_chat_template(
                        rp,
                        tokenize=False,
                        add_generation_prompt=True,
                        )
            # except:
            #     read_prompt = messages_to_string(rp) # handling ministral 
            probe_data.append(
            {
                "read_prompt": read_prompt,
                "dialog": BASE_DIALOG + question,
            }
            
        )
        batch = lqa_tokenize(
            probe_data,
            tokenizer,
            name=args.target_model_name,
            generate=generate,
            mask_type=args.truncate if args.truncate != "none" else None,
            modify_chat_template=args.modify_chat_template,
        )
        result = latent_qa(
            batch,
            target_model,
            decoder_model,
            module_read[0],
            module_write[0],
            tokenizer,
            shift_position_ids=False,
            generate=generate,
            cache_target_model_grad=False,
            sae=sae,
            return_sae_latent=save_sae_dist,
        )
        if save_sae_dist:
            temp, sae_latents = result
            # sae_latents: list of (B, L, d_sae) per layer; we use one layer
            lat = sae_latents[0].detach().cpu()
            for b, global_idx in enumerate(batch_indices):
                # average over sequence: (L, d_sae) -> (d_sae,) per sample
                latent_b = lat[b]
                seq_len = latent_b.shape[0]
                latent_mean = latent_b.sum(dim=0) / seq_len
                latent_str = [f"{x:.6g}" for x in latent_mean.tolist()]
                path = os.path.join("controls", f"sae_distribution_{global_idx}.json")
                with open(path, "w") as f:
                    json.dump({"index": global_idx, "latent": latent_str}, f)
        else:
            temp = result
        out += temp

    responses_data = []
    if generate:
        for i in range(len(out)):
            print(tokenizer.decode(out[i]))
            prompt, completion = clean_text(tokenizer.decode(out[i]))
            # print(f"[PROMPT]: {questions[i % len(questions)][0]['content']}")
            print('')
            print(f"[COMPLETION]: {completion}")
            print('')
            print(f"[GT]: {questions[i][-1]['content']}")
            print("#" * 80)
            
            responses_data.append({
                "index": i,
                "response": completion.strip(),
                "input_prompt": eval(prompt)['content'].strip(),
                "ground_truth": questions[i][-1]['content'].strip()
            })
        if args.save_name != "":
            with open(f"controls/{args.save_name}.jsonl", "w") as f:
                json.dump(responses_data, f, indent=2)
    return responses_data, out, batch

def main(**kwargs):
    from lit.configs.interpret_config import interpret_config
    args = interpret_config()
    update_config(args, **kwargs)
    tokenizer = get_tokenizer(args.target_model_name)
    set_ids = None
    if args.eval_qa.find('paraNMT')!=-1:
        read_prompts, QAs = get_paraNMT_text(args, tokenizer, False)
    if args.eval_qa.find('quora')!=-1:
        read_prompts, QAs = get_quora_text(args, tokenizer, False)
    if args.eval_qa.find('nmt')!=-1:
        read_prompts, QAs = get_nmt_text(args, tokenizer, False)

    dialogs = read_prompts
    questions = QAs

    
    decoder_model = get_model(
        model_name=args.target_model_name,
        tokenizer=tokenizer,
        load_peft_checkpoint=args.decoder_model_name,
        device="cuda:0",
    )
    target_model = get_model(args.target_model_name, tokenizer=tokenizer, device="cuda:1")

    sae = None
    if getattr(args, "use_sae", False):
        ckpt_dir = getattr(args, "sae_checkpoint", "") or args.decoder_model_name
        if ckpt_dir:
            sae = load_sae(
                ckpt_dir,
                "cuda:0",
                args=args,
                hidden_size=decoder_model.config.hidden_size,
            )
            if sae is not None:
                sae.eval()
                print(f"Loaded SAE from {ckpt_dir}/relusae.pt for inference.")
            else:
                print(f"No relusae.pt found in {ckpt_dir}; running without SAE.")

    print(dialogs)
    interpret(target_model, decoder_model, tokenizer, dialogs, questions, args, sae=sae)

if __name__ == "__main__":
    fire.Fire(main)
