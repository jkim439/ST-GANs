import os
from utils import show_all_variables
from utils import check_folder
import argparse
from PIL import Image
import numpy as np

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--iterations', type=int, default=2000, help='The number of iterations to run')
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--group', type=str, default='current',
                        help='Group')
    parser.add_argument('--resume', type=int, default=1, help='To Resume Training')
    parser.add_argument('--train', type=int, default=0, help='Training (1) or Testing (0)')
    parser.add_argument('--ckpt_path', type=str, default='.',
                        help='Checkpoint Path')
    parser.add_argument('--label', type=str, default='1',
                        help='Hemo type.')
    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.iterations >= 1, 'number of iterations must be larger than or equal to one'

    # --batch_size
    assert args.batch_size >= 1, 'batch size must be larger than or equal to one'

    # --batch_size
    assert args.resume == 0 or args.resume == 1, 'Resume must be 0 or 1'
    return args


"""main"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    num_gpus = int(len(args.gpu_id) / 2.) + 1
    from GAN5 import WGAN_GP

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = WGAN_GP(sess,
                        iterations=args.iterations,
                        batch_size=max(1, args.train * args.batch_size),
                        checkpoint_dir=args.checkpoint_dir,
                        result_dir=args.result_dir,
                        log_dir=args.log_dir,
                        group=args.group,
                        resume=args.resume,
                        train=args.train,
                        gpus=num_gpus)

        model.build_model()

        if args.train:
            # show network architecture
            show_all_variables()

            model.train()
            print(" [*] Training finished!")
        else:
            if model.load_ckpt(checkpoint_path='./checkpoint'):  # Checks if checkpoint (saved model exists and loads it
                path = "/home/jkim/NAS/members/mkarki/STGAN/brain"
                palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128]]
                random_label = True
                all_fg = []
                # load corresponding foreground mask/ texture
                for i in range(1, 6):
                    foreground = np.load("{0}/mask_{1}_train_256_all.npy".format(path, i))
                    all_fg.append(foreground)

                imgs = os.listdir('inference/background_imgs')
                for img_name in imgs:
                    bg = Image.open('inference/background_imgs/' + img_name)

                    if random_label:
                        label = np.random.randint(1, 6)
                    else:
                        label = int(args.label)
                    rnd = np.random.randint(0, all_fg[label - 1].shape[0])
                    bg = bg.resize((256, 256), Image.ANTIALIAS)
                    bg = np.array(bg)

                    fg = all_fg[label - 1][rnd, :, :, :]

                    img = model.inference(fg / 255.,
                                          np.reshape(bg / 255., (1, 256, 256, 1)))  # EDIT this for different model
                    img = np.squeeze(img[0]) * 255
                    # mask = np.clip((img - bg),0,255).astype(np.uint8)
                    img_name = img_name.replace('NORM', 'FAKE')

                    # Write Images to File
                    j = Image.fromarray(img.astype(np.uint8))
                    j = j.resize((512, 512), Image.ANTIALIAS)

                    j.save('inference/output_imgs5/' + img_name, 'PNG')
                    # j = Image.fromarray(mask)
                    # j = j.resize((512, 512), Image.ANTIALIAS)
                    # j.putpalette(palette[0] + palette[label])
                    # j.save('inference/output_lbls5/' + img_name, 'PNG')
                print(" [*] Inference finished!")
            else:
                print(" [!] Inference failed! No checkpoint in path!")


if __name__ == '__main__':
    main()


