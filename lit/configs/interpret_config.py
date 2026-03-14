from dataclasses import dataclass

@dataclass
class interpret_config:
    # Model
    decoder_model_name: str = ""
    target_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # SAE: load from training checkpoint for full structure (LatentQA + SAE) inference
    use_sae: bool = False
    # Directory containing relusae.pt; if "", use decoder_model_name (same dir as LatentQA)
    sae_checkpoint: str = ""
    # For legacy relusae.pt (state_dict only); ignored if checkpoint has config
    sae_type: str = "relu"
    sae_dim: int = 16384
    topk_percent: float = 0.005

    # This should match your training setup; these defaults are from our setup
    min_layer_to_read: int = 15
    max_layer_to_read: int = 16
    num_layers_to_read: int = 1
    num_layers_to_sample: int = 1
    layer_to_write: int = 0
    module_setup: str = "read-vary_write-fixed_n-fixed"
    
    # Other args
    seed: int = 42
    batch_size: int = 16
    modify_chat_template: bool = True
    truncate: str = "none"
    save_name: str = ""
    prompt: str = ""
    # Save SAE latent distribution per sample to controls/ (numbers as strings to save space)
    save_sae_distribution: bool = False

    # Evaluation Dataset # data/NegotiationToM/test.json
    eval_qa: str = "/home/z5517269/Semantic-SAE/data/paraNMT/test.txt"