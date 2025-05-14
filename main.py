import os
import gc
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("generated", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated", StaticFiles(directory="generated"), name="generated")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "gif_url": None})

@app.get("/generate", response_class=HTMLResponse)
def generate_video(request: Request, prompt: str = ""):
    if not prompt:
        return templates.TemplateResponse("index.html", {"request": request, "gif_url": None})

    gc.collect()
    torch.cuda.empty_cache()

    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out")
    pipe.scheduler = DDIMScheduler.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1
    )

    pipe.to("cpu")
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, bad anatomy",
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=torch.Generator().manual_seed(42),
    )

    gif_path = os.path.join("generated", f"video_{torch.randint(1000,9999,(1,)).item()}.gif")
    export_to_gif(output.frames[0], gif_path)

    return templates.TemplateResponse("index.html", {"request": request, "gif_url": "/" + gif_path})