import cv2
import torch
import argparse
import torch 
from diffusers import AutoencoderKLWan, WanPipeline 
from diffusers.utils import export_to_video, load_video
import gc
from nash_cash import NashCaSHConfig

parser = argparse.ArgumentParser(description='Choose between cache and nocache')
parser.add_argument('--mode', type=str, choices=['cache', 'nocache'], default='nocache', help="Specify either 'cache' or 'nocache' mode")
parser.add_argument('--target_height', type=int, default=1088, help="Target height for Your Super Resolution Video")
parser.add_argument('--target_width', type=int, default=1920, help="Target width for Your Super Resolution Video")
parser.add_argument('--base_height', type=int, default=480, help="Native-height first-stage generation size")
parser.add_argument('--base_width', type=int, default=832, help="Native-width first-stage generation size")
parser.add_argument('--num_frames', type=int, default=81, help="Number of frames for first-stage generation")
parser.add_argument('--guidance_scale', type=float, default=5.0, help="Classifier-free guidance scale")
parser.add_argument('--strength', type=float, default=0.7, help="SDEdit noise strength for high-resolution refinement")
parser.add_argument('--flow_shift', type=float, default=9.0, help="Flow-shift value for high-resolution refinement")
parser.add_argument('--cache_steps', type=int, default=2, help="Refresh interval for cached full-branch cross-attention")
parser.add_argument('--output', type=str, default='Nash_CaSH_Old_man.mp4', help="Output video path")
parser.add_argument('--prompt', type=str, default="A realistic close-up of an elderly man with gray hair and a thick gray beard, wearing a light-colored shirt. His head is slightly lowered. The camera zooms from full body to close-up, highlighting detailed facial wrinkles, skin texture, forehead lines, eye bags, and beard strands. High resolution, cinematic lighting, sharp details.", help="Text prompt")
parser.add_argument('--prompt_file', type=str, default=None, help="Optional text file containing the prompt")
parser.add_argument('--negative_prompt', type=str, default="repeating patterns, Blurry face, low detail, distorted features, extra limbs, cartoon style, smooth plastic skin, low resolution, flat colors, lack of texture", help="Negative prompt")
parser.add_argument('--negative_prompt_file', type=str, default=None, help="Optional text file containing the negative prompt")
parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducible visual comparisons")
parser.add_argument('--nash_cash', action=argparse.BooleanOptionalAction, default=True, help="Enable Nash-CaSH cross-attention bargaining")
parser.add_argument('--nash_floor', type=float, default=0.65, help="Minimum full-branch share in Nash-CaSH")
parser.add_argument('--nash_ceiling', type=float, default=0.98, help="Maximum full-branch share in Nash-CaSH")
parser.add_argument('--nash_temperature', type=float, default=1.0, help="Temperature for Nash-CaSH payoff allocation")
parser.add_argument('--nash_full_prior', type=float, default=1.25, help="Prior payoff multiplier for global full-attention branch")
parser.add_argument('--nash_window_prior', type=float, default=1.0, help="Prior payoff multiplier for local window-attention branch")
args = parser.parse_args()

if args.mode == 'cache':
    from cache.transformer_wan import WanTransformer3DModel
    from cache.pipeline_wan_video2video import WanVideoToVideoPipeline
    from cache.attention_processor_ import init_mask_flex, WanFlexAttnProcessor_, WanCrossAttnProcessor
elif args.mode == 'nocache':
    from nocache.transformer_wan import WanTransformer3DModel
    from nocache.pipeline_wan_video2video import WanVideoToVideoPipeline
    from nocache.attention_processor_ import init_mask_flex, WanFlexAttnProcessor_, WanCrossAttnProcessor

assert args.target_width <= 1920 and args.target_height <= 1088, \
    "Please use 14B model at higher resolutions for better results"

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

pipe_t2v = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
# You can accelerate by change the following line with pipe_t2v.enable_model_cpu_offload(), but require more memory
pipe_t2v.enable_sequential_cpu_offload()
pipe_t2v.vae.enable_tiling()

num_frames = args.num_frames
base_height = args.base_height
base_width = args.base_width
# If you want to use the 14B models, please remember to change the base_width to 1280, and base_height to 720.

prompt = open(args.prompt_file, "r", encoding="utf-8").read().strip() if args.prompt_file else args.prompt
negative_prompt = (
    open(args.negative_prompt_file, "r", encoding="utf-8").read().strip()
    if args.negative_prompt_file
    else args.negative_prompt
)
generator = None
if args.seed is not None:
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

output = pipe_t2v(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=base_height,
    width=base_width,
    num_frames=num_frames,
    guidance_scale=args.guidance_scale,
    generator=generator,
).frames[0]

# You can export the base video to compare native and high-resolution refinement.
# export_to_video(output, "base_video.mp4")

output = [cv2.resize(item, (args.target_width, args.target_height), interpolation=cv2.INTER_LINEAR) for item in output]

del pipe_t2v
gc.collect()
torch.cuda.empty_cache()

pipe_v2v = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe_v2v.transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16) 
# You can accelerate by change the following line with pipe_v2v.enable_model_cpu_offload(), but require more memory
pipe_v2v.enable_sequential_cpu_offload()
pipe_v2v.vae.enable_tiling()

init_mask_flex(
    num_frames=1 + (num_frames - 1) // 4, 
    height=args.target_height // 16, 
    width=args.target_width // 16,
    d_h=base_height // 16 // 2, 
    d_w=base_width // 16 // 2, 
    device='cuda',
    # if you use other torch version, the memory usage would make a big difference, 
    # but you can specify the device to cpu to avoid this problem,
    # which would make the speed slower for around 5s
)

pipe_v2v.scheduler.config.flow_shift = args.flow_shift

del init_mask_flex

nash_config = NashCaSHConfig(
    enabled=args.nash_cash,
    floor=args.nash_floor,
    ceiling=args.nash_ceiling,
    temperature=args.nash_temperature,
    full_prior=args.nash_full_prior,
    window_prior=args.nash_window_prior,
)

attn_processors = {}
for k in pipe_v2v.transformer.attn_processors.keys():
    if 'attn2' in k:
        attn_processors[k] = WanCrossAttnProcessor(nash_config=nash_config)
    else:
        attn_processors[k] = WanFlexAttnProcessor_()
pipe_v2v.transformer.set_attn_processor(attn_processors)

pipe_kwargs = dict(
    video=output,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=args.target_height,
    width=args.target_width,
    guidance_scale=args.guidance_scale,
    strength=args.strength,
    generator=generator,
)
if args.mode == 'cache':
    pipe_kwargs["cache_steps"] = args.cache_steps

output = pipe_v2v(**pipe_kwargs).frames[0]

export_to_video(output, args.output, fps=15)
