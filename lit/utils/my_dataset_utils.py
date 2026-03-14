from torch.utils.data import Dataset

class MyLatentQADataset(Dataset):
    def __init__(
        self,
        tokenizer,
        read_prompts,
        QAs,
        is_dialog = True,
    ):
        self.BD = [
            # {
            #     "role": "assistant",
            #     "content": "Sure, I've analyzed the given sentence",
            # }
            {
                "role": "assistant",
                "content": "Sure, I'm an expert in translation",
            }
        ]
        self.tokenizer = tokenizer
        self.read_prompts = read_prompts
        self.QAs = QAs
        self.is_dialog = is_dialog
        self.lengths = []
        if self.is_dialog:
            for rp, qa in zip(read_prompts, QAs):
                self.lengths.append(sum([len(item['content']) for item in rp])+len(qa[0]['content'])+len(qa[1]['content']))
        else:
            for rp, qa in zip(read_prompts, QAs):
                self.lengths.append(len(rp)+len(qa[0]['content'])+len(qa[1]['content']))


    def __len__(self):
        return len(self.QAs)

    def __getitem__(self, idx):
        read_prompt = self.read_prompts[idx]
        qa_dialog = self.QAs[idx]
        try:
            read_prompt = self.tokenizer.apply_chat_template(
                        read_prompt, tokenize=False, add_generation_prompt=False
                    )
        except:
            read_prompt = messages_to_string(read_prompt) # handling ministral 
        return {"read_prompt": read_prompt, "dialog": self.BD + qa_dialog, "mask_type":"user"}



def batch_index_generator(dataset_size, batch_size):
    current_index = 0
    while current_index < dataset_size:
        end_index = min(current_index + batch_size, dataset_size)
        yield list(range(current_index, end_index))
        current_index = end_index

def get_paraNMT_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    with open(data_path, 'r') as fin:
        for line in fin:
            sentence1, sentence2 = line.strip().split('\t')
            read_prompts.append([{"role": "user", "content": sentence1}])
            question = "What does this question talk about?"
            answer = sentence2
            QAs.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ])

    assert len(QAs)==len(read_prompts)
    return read_prompts, QAs

def get_quora_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    with open(data_path, 'r') as fin:
        for idx, line in enumerate(fin):
            if idx==0:
                continue
            sentence1, sentence2 = line.strip().split(',')[3:5]
            read_prompts.append([{"role": "user", "content": sentence1}])
            question = "Rewrite another version the given sentence."
            answer = sentence2
            QAs.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ])

    assert len(QAs)==len(read_prompts)
    return read_prompts, QAs

def get_nmt_text(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    
    with open(data_path, 'r') as fin:
        for idx, line in enumerate(fin):
            if len(line.strip().split('\t'))==3:
                [src, lang, trt] = line.strip().split('\t')
                # if lang == 'fa_IR':
                read_prompts.append([{"role": "user", "content": src}])
                question = f"Translate the given sentence into {lang} Lang"
                answer = trt
                QAs.append([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                        ])

    assert len(QAs)==len(read_prompts)
    return read_prompts, QAs

def get_quora_dataset(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    with open(data_path, 'r') as fin:
        for idx, line in enumerate(fin):
            if idx==0:
                continue
            sentence1, sentence2 = line.strip().split(',')[3:5]
            read_prompts.append([{"role": "user", "content": sentence1}])
            question = "Rewrite another version the given sentence."
            answer = sentence2
            QAs.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ])

    assert len(QAs)==len(read_prompts)
    return MyLatentQADataset(
        tokenizer,
        read_prompts,
        QAs
    )


def get_paraNMT_dataset(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    with open(data_path, 'r') as fin:
        for line in fin:
            sentence1, sentence2 = line.strip().split('\t')
            read_prompts.append([{"role": "user", "content": sentence1}])
            question = "Write a paraphrased version of the given sentence."
            answer = sentence2
            QAs.append([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                    ])

    assert len(QAs)==len(read_prompts)
    return MyLatentQADataset(
        tokenizer,
        read_prompts,
        QAs
    )

def get_nmt_dataset(train_config, tokenizer, train=True):
    data_path = train_config.train_qa if train else train_config.eval_qa
    read_prompts, QAs = [], []
    with open(data_path, 'r') as fin:
        for idx, line in enumerate(fin):
            # if idx==0:
            #     continue
            if len(line.strip().split('\t'))==3:
                [src, lang, trt] = line.strip().split('\t')
                read_prompts.append([{"role": "user", "content": src}])
                question = f"Translate the given sentence into {lang} Lang"
                answer = trt
                QAs.append([
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                        ])

    assert len(QAs)==len(read_prompts)
    return MyLatentQADataset(
        tokenizer,
        read_prompts,
        QAs
    )


def get_my_dataset(train_config, tokenizer, train=True):
    if train_config.train_qa.find('paraNMT')!=-1:
        return get_paraNMT_dataset(train_config, tokenizer, train)
    if train_config.train_qa.find('quora')!=-1:
        return get_quora_dataset(train_config, tokenizer, train)
    if train_config.train_qa.find('nmt')!=-1:
        return get_nmt_dataset(train_config, tokenizer, train)

    else:
        return Exception('There is NO such dataset!')