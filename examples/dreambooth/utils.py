from PIL import Image
import jittor as jt
import json
from JDiffusion.utils import randn_tensor


modules = ['down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q',
'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k',
'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v',
'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0',
'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q',
'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k',
'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v',
'down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0',
'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q',
'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k',
'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v',
'down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0',
'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q',
'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k',
'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v',
'down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0',
'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q',
'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k',
'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v',
'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0',
'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q',
'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k',
'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v',
'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0',
'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q',
'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k',
'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v',
'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0',
'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q',
'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k',
'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v',
'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0',
'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q',
'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k',
'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v',
'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0',
'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q',
'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k',
'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v',
'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0',
'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q',
'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k',
'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v',
'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0',
'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q',
'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k',
'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v',
'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0',
'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q',
'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k',
'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v',
'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0',
'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q',
'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k',
'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v',
'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0',
'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q',
'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k',
'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v',
'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0',
'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q',
'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k',
'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v',
'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0',
'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q',
'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k',
'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v',
'up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0',
'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q',
'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k',
'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v',
'up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0',
'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q',
'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k',
'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v',
'up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0',
'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q',
'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k',
'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v',
'up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0',
'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q',
'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k',
'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v',
'up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0',
'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q',
'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k',
'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v',
'up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0',
'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q',
'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k',
'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v',
'up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0',
'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q',
'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k',
'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v',
'up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0',
'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q',
'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k',
'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v',
'up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0',
'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q',
'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k',
'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v',
'up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0',
'mid_block.attentions.0.transformer_blocks.0.attn1.to_q',
'mid_block.attentions.0.transformer_blocks.0.attn1.to_k',
'mid_block.attentions.0.transformer_blocks.0.attn1.to_v',
'mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0',
'mid_block.attentions.0.transformer_blocks.0.attn2.to_q',
'mid_block.attentions.0.transformer_blocks.0.attn2.to_k',
'mid_block.attentions.0.transformer_blocks.0.attn2.to_v',
'mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0',
]


id2style_old = {'00':'{obj} of neon style','01':'an oil painting of {obj}','02':'{obj} of v02 style','03':'a drawing of {obj}','04':'{obj} of cloud style',
            '05':'{obj} of v05 style','06':'{obj} of chinese paper cutout style','07':'a drawing of {obj}','08':'{obj} of v08 style','09':'watercolor painting of {obj}',
            '10':'a drawing of {obj}','11':'{obj} of v11 style','12':'{obj} of lego bricks style','13':'{obj} of watercolor painting style','14':'{obj} of v14 style', 
            '15':'{obj} of v15 style','16':'{obj} in pixel art style','17':'a watercolor painting of {obj}', '18':'a drawing of {obj}', '19':'{obj} of lego bricks style',
            '20':'a drawing of {obj}','21':'a paper cut of {obj}', '22':'a drawing of {obj}', '23':'a drawing of {obj}','24':'{obj} of v24 style','25':'pixel art {obj}',
            '26':'{obj} of origami style','27':'{obj} in v27 style'}

id2style = {'00':'a {obj} of neon style',
'01':'an oil painting of a {obj}',
'02':'a {obj} of v02 style',
'03':'a drawing of a {obj}',
'04':'a {obj} of cloud style',
'05':'a {obj} of v05 style',
'06':'a {obj} of chinese paper cutout style',
'07':'a drawing of a {obj}',
'08':'a {obj} of v08 style',
'09':'watercolor painting of a {obj}',
'10':'a drawing of a {obj}',
'11':'a {obj} of v11 style',
'12':'a {obj} of lego bricks style',
'13':'a {obj} of watercolor painting style',
'14':'a {obj} of v14 style',
'15':'a {obj} of v15 style',
'16':'a {obj} in pixel art style',
'17':'a watercolor painting of a {obj}',
'18':'a drawing of a {obj}',
'19':'a {obj} of lego bricks style',
'20':'a drawing of a {obj}',
'21':'a paper cut of a {obj}',
'22':'a drawing of a {obj}',
'23':'a drawing of a {obj}',
'24':'a {obj} of v24 style',
'25':'pixel art a {obj}',
'26':'a {obj} of origami style',
'27':'a {obj} in v27 style'}   

id2style_v2 = {'00':'a {obj} in neon style',
'01':'an oil painting of a {obj}',
'02':'a {obj} in v02 style',
'03':'a drawing of a {obj}',
'04':'a {obj} in v04 style',
'05':'a {obj} in v05 style',
'06':'a {obj} in chinese paper cutout style',
'07':'a drawing of a {obj}',
'08':'a {obj} in v08 style',
'09':'watercolor painting of a {obj}',
'10':'a drawing of a {obj}',
'11':'a {obj} in v11 style',
'12':'a {obj} in v12 style',
'13':'a {obj} in v13 style',
'14':'a {obj} in v14 style',
'15':'a {obj} in v15 style',
'16':'a {obj} in pixel art style',
'17':'a watercolor painting of a {obj}',
'18':'a drawing of a {obj}',
'19':'a {obj} in lego bricks style',
'20':'a drawing of a {obj}',
'21':'a paper cut in a {obj}',
'22':'a drawing of a {obj}',
'23':'a drawing of a {obj}',
'24':'a {obj} in v24 style',
'25':'a {obj} in v25 style',
'26':'a {obj} in origami style',
'27':'a {obj} in v27 style'}  

neg_p = """
Worst quality,
 Normal quality,
 Low quality,
 Low res,
 Blurry,
 Jpeg artifacts,
 Grainy,
 text,
 logo,
 watermark,
 banner,
 extra digits,
 signature,
 man,
 woman,
 Cropped,
 Out of frame,
 Out of focus.  
"""

# Useful function for later
def load_image(path, size=(512,512), is_grey=False):
    if not is_grey:
        img = Image.open(path).convert("RGB")
    else:
        img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size)
    return img


def to_tensor(img):
    totensor =jt.transform.ToTensor()
    return jt.array(totensor(img))

def init_latent_jt(model, seed=None, dtype=jt.float32):
    scale_factor = model.vae_scale_factor # 8
    sample_size = 64 # after vae's dimension
    latent_dim = model.unet.config.in_channels # in xl, it is 4.

    height = sample_size * scale_factor
    width = sample_size * scale_factor

    shape = (1, latent_dim, height // scale_factor, width // scale_factor)

    latent = randn_tensor(shape, seed, dtype=dtype)

    return latent
