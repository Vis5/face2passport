import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys
import random
import pretrained_networks

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

def main():
    _G, _D, Gs = pretrained_networks.load_networks("results/00005-stylegan2-500_128_passport-1gpu-config-f/network-snapshot-000361.pkl")
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 4

    for img_name in [f for f in os.listdir('generated_images') if f[0] not in '._']:
        if not img_name.endswith('.npy'):
            continue
        path = os.path.join('generated_images', img_name)
        input_img = np.load(path)

        seed = random.randint(1,999)
        z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:])])
        w = Gs.components.mapping.run(z, None)
        w = w_avg + (w + w_avg) * 1.0
        passport = Gs.components.synthesis.run(w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
        w = w[0]
        col_styles = _parse_num_range('0-6')
        input_img[col_styles] = w[col_styles]
        image = Gs.components.synthesis.run(input_img[np.newaxis], **Gs_syn_kwargs)[0]
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('output_images/%s.png' % (os.path.splitext(img_name)[0])))

if __name__ == "__main__":
    main()
