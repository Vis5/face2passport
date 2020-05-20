'''
Written by Tuvshinbayar Dashdavaa
2020 Apr
'''
import os
import sys
import argparse
import shutil
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
import dataset_tool

from keras.utils import get_file
from training import dataset
from training import misc
from preprocessing.face_alignment import image_align
from preprocessing.landmarks_detector import LandmarksDetector


def project_image(proj, src_file, dst_dir, tmp_dir):
    # Translate into .tfrecords
    data_dir = '%s/dataset' % tmp_dir
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    image_dir = '%s/images' % data_dir
    tfrecord_dir = '%s/tfrecords' % data_dir
    os.makedirs(image_dir, exist_ok=True)
    shutil.copy(src_file, image_dir + '/')
    dataset_tool.create_from_images(tfrecord_dir, image_dir, shuffle=0)
    dataset_obj = dataset.load_dataset(
        data_dir=data_dir, tfrecord_dir='tfrecords',
        max_label_size=0, repeat=False, shuffle_mb=0
    )

    # Project image into latent space and find the optimized results
    print('Projecting image "%s"...' % os.path.basename(src_file))
    images, _labels = dataset_obj.get_minibatch_np(1)
    images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
    proj.start(images)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
    print('\r%-30s\r' % '', end='', flush=True)

    # Save generated files
    os.makedirs(dst_dir, exist_ok=True)
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.png')
    misc.save_image_grid(proj.get_images(), filename, drange=[-1,1])
    filename = os.path.join(dst_dir, os.path.basename(src_file)[:-4] + '.npy')
    np.save(filename, proj.get_dlatents()[0])


def main():
    # Align 1024x1024 face image from raw images using face shape predictor
    landmarks_detector = LandmarksDetector('models/shape_predictor_68_face_landmarks.dat')
    for img_name in [f for f in os.listdir('raw_images') if f[0] not in '._']:
        raw_img_path = os.path.join('raw_images', img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join('aligned_images', face_img_name)
            os.makedirs('aligned_images', exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)

    # Load the model
    network_pkl = "results/00005-stylegan2-500_128_passport-1gpu-config-f/network-snapshot-000361.pkl"
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector(
        vgg16_pkl             = "models/vgg16_zhang_perceptual.pkl",
        num_steps             = 1000,
        initial_learning_rate = 0.1,
        initial_noise_factor  = 0.05,
        verbose               = False
    )
    proj.set_network(Gs)
    src_files = sorted([os.path.join('aligned_images', f) for f in os.listdir('aligned_images') if f[0] not in '._'])
    for src_file in src_files:
        project_image(proj, src_file, 'generated_images', '.stylegan2-tmp')
        shutil.rmtree('.stylegan2-tmp')

if __name__ == "__main__":
    main()