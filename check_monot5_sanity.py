import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = 'cramraj8/duqgen-monot5-3b-robust04-1k'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32)
    model.to(DEVICE)
    model.eval()

    # Known relevant pair from Robust04 (qid 301, doc FBIS3-10082 is relevant len=497)
    query = "international organized crime"
    # Just a dummy relevant text
    doc_relevant = "International organized crime is a major threat to global security. Criminal organizations operate across borders."
    doc_irrelevant = "The banana is an edible fruit – botanically a berry – produced by several kinds of large herbaceous flowering plants in the genus Musa."

    pairs = [
        (query, doc_relevant),
        (query, doc_irrelevant)
    ]

    true_token_id = tokenizer.encode('true')[0]
    false_token_id = tokenizer.encode('false')[0]
    print(f"True token ID: {true_token_id}, False token ID: {false_token_id}")

    prompts = [f"Query: {q} Document: {d} Relevant:" for q, d in pairs]
    
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
        decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]] * len(prompts)).to(DEVICE)
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits # (B, 1, V)
        
        # Check logits for true/false
        true_logits = logits[:, 0, true_token_id]
        false_logits = logits[:, 0, false_token_id]
        
        print(f"True Logits: {true_logits}")
        print(f"False Logits: {false_logits}")
        
        probs = torch.softmax(logits[:, 0, :], dim=-1)
        true_probs = probs[:, true_token_id]
        
        print(f"True Probs (vocab-wide): {true_probs}")
        
        # Binary softmax
        binary_logits = torch.stack([false_logits, true_logits], dim=1)
        binary_probs = torch.softmax(binary_logits, dim=1)
        print(f"Binary Probs (False, True): \n{binary_probs}")

if __name__ == "__main__":
    main()
