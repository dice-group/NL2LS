class CustomTask:
    """
    In this task setup the model will learn to predict the output based on a given input.
    The model will use divider_token to separate input from output for learning.
    """
    
    default_pad_token = "[PAD]"
    default_eos_token = "</s>"
    default_bos_token = "<s>"
    default_unk_token = "<unk>"
    
    def __init__(self) -> None:
        super().__init__()
        self.input_template = "Translate text into Link Specification:\n{input}\n"
    
    def before_train(self, model, tokenizer):
        self.add_new_tokens(model, tokenizer)

    def add_new_tokens(self, model, tokenizer):
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = self.default_pad_token
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = self.default_eos_token
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = self.default_bos_token
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = self.default_unk_token
        
        tokenizer.add_special_tokens(special_tokens_dict)
        
        model.resize_token_embeddings(len(tokenizer))
            

    def build_source_and_target(self, list_data_dict, tokenizer):
        sources = [
            self.input_template.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        return sources, targets