import os
import sys

# Force aggressive debugging env vars
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
os.environ['NCCL_DEBUG'] = 'INFO'

try:
    from vllm import LLM, SamplingParams
    print("vLLM imported successfully.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def run_test():
    print("Initializing LLM Engine...")
    try:
        # Use a tiny model to avoid OOM, just testing checking kernels/driver
        llm = LLM(model="HuggingFaceM4/tiny-random-LlamaForCausalLM", enforce_eager=True)
        print("LLM Engine initialized!")
        
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated: {generated_text!r}")
            
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
