import sys
import os
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import torch
import threading

# Path to dual_system — override with DUAL_SYSTEM_PATH env var if needed.
# Default: assumes dual_system/ is a sibling directory of this repo.
# e.g. if both repos are cloned side-by-side:
#   ~/projects/dual_system/
#   ~/projects/smallville-bender/   ← this repo
_default_dual_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dual_system")
)
DUAL_SYSTEM_PATH = os.environ.get("DUAL_SYSTEM_PATH", _default_dual_path)
sys.path.append(DUAL_SYSTEM_PATH)

from chat_continuous import (
    load_v2_system, 
    generate_streaming, 
    continuous_learning_update
)
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator

app = FastAPI(title="BENDER Backend API")

gpu_lock = threading.Lock()

# Global states
model = None
tokenizer = None
optimizer_sidecar = None
optimizer_base = None
optimizer_sparse = None
epistemic_router = None
bch_consolidator = None
device = "cuda"

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, optimizer_sidecar, optimizer_base, optimizer_sparse
    global epistemic_router, bch_consolidator, device
    
    backbone_id = os.environ.get(
        "BENDER_BACKBONE", "Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED"
    )
    sidecar_path = os.environ.get(
        "BENDER_SIDECAR", os.path.join(DUAL_SYSTEM_PATH, "sidecar_step500.pt")
    )
    
    print("Loading Dual-System V2 backend...")
    model, tokenizer = load_v2_system(backbone_id, sidecar_path)
    
    # [CRITICAL SHIELD] Upcast sidecar parameters to FP32. AdamW `v = g^2` natively 
    # overflows FP16 max value (65504) causing irreversible NaN logic collapse globally.
    model.geo_processor.to(torch.float32)
    model.ebm_critic.to(torch.float32)
    if getattr(model, 'latent_planner', None) is not None:
        model.latent_planner.to(torch.float32)
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        model.fiber_bundle.to(torch.float32)
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj.to(torch.float32)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias.to(torch.float32)
    
    sidecar_params = list(model.geo_processor.parameters()) + list(model.ebm_critic.parameters())
    if getattr(model, 'fiber_proj', None) is not None:
        sidecar_params += list(model.fiber_proj.parameters())
        
    optimizer_groups = []
    if sidecar_params:
        optimizer_groups.append({'params': sidecar_params, 'weight_decay': 0.01})
        
    per_user_params = []
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        per_user_params.append(model.fiber_bundle.fiber_store.fiber_vectors)
        
    if per_user_params:
        optimizer_groups.append({'params': per_user_params, 'weight_decay': 0.0})
        
    optimizer_sidecar = torch.optim.AdamW(optimizer_groups, lr=5e-5)
    
    if getattr(model, 'sparse_bias', None) is not None:
        optimizer_sparse = torch.optim.SparseAdam(list(model.sparse_bias.parameters()), lr=1e-4)
        
    base_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer_base = torch.optim.AdamW(base_params, lr=1e-6, weight_decay=0.0)
    
    epistemic_router = EpistemicRouter(energy_threshold_init=5.0, tau_e=0.5, eta_base=0.01, eta_user=0.05).to(device)
    epistemic_router.train()
    
    bch_consolidator = BCHConsolidator(eta=0.05, order=2)
    print("BENDER API initialized.")

class GenerateRequest(BaseModel):
    prompt: str
    user_id: int

class InjectMemoryRequest(BaseModel):
    prompt: str
    target_response: str
    reward: float
    user_id: int

@app.post("/generate")
async def generate(req: GenerateRequest):
    def _do_generate():
        try:
            with gpu_lock:
                prompt_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
                tokens = []
                for token_id in generate_streaming(model, tokenizer, prompt_ids, user_id=req.user_id):
                    tokens.append(token_id)
                return tokenizer.decode(tokens, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "BENDER SYSTEM: CUDA Out of Memory. Generation fallback triggered."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"BENDER SYSTEM: Generation Error - {str(e)}"

    generated_text = await asyncio.to_thread(_do_generate)
    return {"generated_text": generated_text}

@app.post("/inject_memory")
async def inject_memory(req: InjectMemoryRequest, background_tasks: BackgroundTasks):
    def _do_update():
        try:
            with gpu_lock:
                prompt_ids = tokenizer(req.prompt, return_tensors="pt").input_ids.to(device)
                target_ids = tokenizer(req.target_response, return_tensors="pt").input_ids.to(device)
                dummy_resp_ids = torch.tensor([[tokenizer.eos_token_id]], device=device)
                
                loss, energy, kappa = continuous_learning_update(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    resp_ids=dummy_resp_ids,
                    reward=req.reward,
                    optimizer_sidecar=optimizer_sidecar,
                    optimizer_base=optimizer_base,
                    epistemic_router=epistemic_router,
                    bch_consolidator=bch_consolidator,
                    user_id=req.user_id,
                    dtype=torch.bfloat16,
                    target_ids=target_ids,
                    optimizer_sparse=optimizer_sparse
                )
                print(f"Memory injected user={req.user_id} | Loss={loss:.3f} kappa={kappa:.3f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"Memory injection OOM for user={req.user_id}. Catching explicitly.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Memory injection error for user={req.user_id}: {str(e)}")
        
    background_tasks.add_task(_do_update)
    return {"status": "Update scheduled in background"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
