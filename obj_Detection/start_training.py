from . import train
import os
import logging
import argparse
from .utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from .utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class Opt:
    def __init__(self, data, epoch, batch_size, img_size, project, name, save_dir):
        self.weights = './obj_Detection/yolov5s.pt'
        self.cfg = './obj_Detection/models/yolov5s.yaml'
        self.data = data
        self.hyp = './obj_Detection/data/hyp.scratch.yaml'
        self.epochs = epoch
        self.batch_size = batch_size
        self.img_size = [img_size, img_size]
        self.rect = False
        self.resume = False
        self.nosave = False
        self.notest = False
        self.noautoanchor = False
        self.evolve = False
        self.bucket = ''
        self.cache_images = False
        self.image_weights = False
        self.device = ''
        self.multi_scale = False
        self.single_cls = False
        self.adam = False
        self.sync_bn = False
        self.local_rank = -1
        self.log_imgs = 16
        #opt.log_artifacts = false
        self.workers = 2
        self.project = project
        self.entity = None
        self.name = name
        self.exist_ok = False
        self.quad = False
        self.linear_lr = False
        self.save_dir = save_dir
        self.total_batch_size = 8
        self.label_smoothing = 0.0


def start_training(data, epoch, batch_size, img_size, project, name, save_dir):
    parser = argparse.ArgumentParser()
    logger = logging.getLogger(__name__)
    # opt = parser.parse_args()
    opt = Opt(data, epoch, batch_size, img_size, project, name, save_dir)

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']
                         ) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # if opt.global_rank in [-1, 0]:
    #     check_requirements()

    # Resume
    if opt.resume:  # resume an interrupted run
        # specified or most recent path
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(
            ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(
                **yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
            opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(
            opt.weights), 'either --cfg or --weights must be specified'
        # extend to 2 sizes (train, test)
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(
            opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # distributed backend
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    try:
        import wandb
    except ImportError:
        wandb = None
        prefix = colorstr('wandb: ')
        logger.info(
            f"{prefix}Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)")
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(
                f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train.train(hyp, opt, device, tb_writer, wandb)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                # final OneCycleLR learning rate (lr0 * lrf)
                'lrf': (1, 0.01, 1.0),
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                # anchors per output grid (0 to ignore)
                'anchors': (2, 2.0, 10.0),
                # focal loss gamma (efficientDet default gamma=1.5)
                'fl_gamma': (0, 0.0, 2.0),
                # image HSV-Hue augmentation (fraction)
                'hsv_h': (1, 0.0, 0.1),
                # image HSV-Saturation augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),
                # image HSV-Value augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                # image perspective (+/- fraction), range 0-0.001
                'perspective': (0, 0.0, 0.001),
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / \
            'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' %
                      opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[
                        0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / \
                        w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng)
                         * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, wandb=wandb)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
