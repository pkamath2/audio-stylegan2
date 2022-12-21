import time
import sys
import copy
import os
import numpy as np
import json
import psutil
import pickle
import torch
from torchsummary import summary

sys.path.insert(0, '../')
from training.dataset import AudioDataset
from util import util
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from training.networks import Generator, Discriminator
from training.loss import StyleGAN2Loss
from metrics import metric_main

def train(config, random_seed):
    start_time = time.time()
    device = util.get_device()
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    config['out_dir'] = util.pick_out_dir(config['out_dir'])
    print('Creating output directory...')
    os.makedirs(config['out_dir'])
    with open(os.path.join(config['out_dir'], 'training_options.json'), 'wt') as f:
        json.dump(config, f, indent=2)
    

    #From Nvidia's repo
    torch.backends.cudnn.benchmark = True    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = True        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                 # Improves training speed.
    grid_sample_gradfix.enabled = True

    print('Loading training set...')
    training_dataset = AudioDataset(config['data_dir'], config['config_file'])
    training_dataset_sampler = misc.InfiniteSampler(dataset=training_dataset, rank=0, num_replicas=1, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_dataset, 
                                                                    sampler=training_dataset_sampler, 
                                                                    batch_size=config['batch_size'], 
                                                                    pin_memory=True, 
                                                                    num_workers=3, 
                                                                    prefetch_factor=2))
    print('Audio Dataset options:', '\n',\
                        'Number of audio files', len(training_dataset),'\n',\
                        'Audio spectrogram shape', training_dataset._raw_shape,\
                        'Label shape:', 'TODO')

    fmaps = 1 if training_dataset.resolution >= 512 else 0.5

    # Construct networks.
    print('Constructing networks...')
    G_kwargs = {
        'z_dim':config['z_dim'],
        'c_dim':0, #TODO: Labels: training_dataset.label_dim
        'w_dim':config['w_dim'],
        'img_resolution':training_dataset.resolution,
        'img_channels':1,
        'mapping_kwargs':{
            'num_layers': config['num_mapping_layers'],
        },
        'synthesis_kwargs':{
            'channel_base': int(fmaps * 32768), #Factor to control number of feature maps for each layer.
            'channel_max': 512, #Maximum number of feature maps in each layer.
            'num_fp16_res': 4,# enable mixed-precision training
            'conv_clamp': 256,# clamp activations to avoid float16 overflow
            'fp16_channels_last':True
        }
    }
    G = Generator(**G_kwargs).requires_grad_(False).to(device) # subclass of torch.nn.Module
    
    D_kwargs = {
            'c_dim':0, #TODO: Labels
            'img_resolution': training_dataset.resolution,
            'img_channels':1,
            'architecture':'resnet', #We use the default config. Generator in skip mode and Discriminator in Residual
            'channel_base':int(fmaps * 32768), #Factor to control number of feature maps for each layer.
            'channel_max':512, #Maximum number of feature maps in each layer.
            'num_fp16_res':4,# enable mixed-precision training
            'conv_clamp':256,# clamp activations to avoid float16 overflow
            'epilogue_kwargs':{
                'mbstd_group_size': min(config['batch_size'], 4)
            },
            'block_kwargs':{},
            'mapping_kwargs':{}
    }
    D = Discriminator(**D_kwargs).train().requires_grad_(False).to(device)
    
    config['ema_kimg'] = config['batch_size'] * 10/32
    G_ema = copy.deepcopy(G).eval()

    # Print network summary tables.
    print('Print network summary tables...')
    z = torch.empty([config['batch_size'], G.z_dim], device=device)
    c = torch.empty([config['batch_size'], G.c_dim], device=device)
    img = misc.print_module_summary(G, [z, c])
    print(img.shape,'-----------------------')
    misc.print_module_summary(D, [img, c])

    # Setup Loss & Optim
    print('Setting up training phases...')
    gamma = 0.0002 * (training_dataset.resolution ** 2) / config['batch_size'] # heuristic formula                            
    loss = StyleGAN2Loss(device=device,
                            G_mapping=G.mapping,
                            G_synthesis=G.synthesis,
                            D=D,
                            augment_pipe=None, # No augmentations applied to Audio
                            r1_gamma=gamma
                        )

    
    phases = []
    # 1. G optim
    G_reg_interval = 4
    G_mb_ratio=G_reg_interval / (G_reg_interval + 1)
    G_opt = torch.optim.Adam(G.parameters(), 
                                lr=config['learning_rate'] * G_mb_ratio,
                                betas=[beta ** G_mb_ratio for beta in [0,0.99]]
                            )
    phases += [dnnlib.EasyDict(name='Gmain', module=G, opt=G_opt, interval=1)]
    phases += [dnnlib.EasyDict(name='Greg', module=G, opt=G_opt, interval=G_reg_interval)]


    # 2. D optim
    D_reg_interval = 16
    D_mb_ratio=D_reg_interval / (D_reg_interval + 1)
    D_opt = torch.optim.Adam(D.parameters(), 
                                lr=config['learning_rate'] * D_mb_ratio,
                                betas=[beta ** D_mb_ratio for beta in [0,0.99]]
                            )
    phases += [dnnlib.EasyDict(name='Dmain', module=D, opt=D_opt, interval=1)]
    phases += [dnnlib.EasyDict(name='Dreg', module=D, opt=D_opt, interval=D_reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        phase.start_event = torch.cuda.Event(enable_timing=True)
        phase.end_event = torch.cuda.Event(enable_timing=True)
    

    # Export sample images.
    print('Exporting sample images...')
    grid_size = None
    grid_z = None
    grid_c = None
    grid_size, images, labels = util.setup_snapshot_image_grid(training_set=training_dataset)
    util.save_image_grid(images, os.path.join(config['out_dir'], 'reals.png'), drange=[0,255], grid_size=grid_size)
    grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(config['batch_size'])
    grid_c = torch.from_numpy(labels).to(device).split(config['batch_size'])
    images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
    util.save_image_grid(images, os.path.join(config['out_dir'], 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)


    # Initialize logs
    print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None

    stats_jsonl = open(os.path.join(config['out_dir'], 'stats.jsonl'), 'wt')
    try:
        import torch.utils.tensorboard as tensorboard
        stats_tfevents = tensorboard.SummaryWriter(config['out_dir'])
    except ImportError as err:
        print('Skipping tfevents export:', err)

    print(f'Training for '+str(config['total_kimg'])+' kimg...')
    print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            batch_size = config['batch_size']
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_size)
            phase_real_c = phase_real_c.to(device).split(batch_size)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_size) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_dataset.get_label(np.random.randint(len(training_dataset))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_size) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_size * 1) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, sync=sync, gain=gain)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param in phase.module.parameters():
                    if param.grad is not None:
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))


        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = config['ema_kimg'] * 1000
            if config['ema_rampup'] is not None:
                ema_nimg = min(ema_nimg, cur_nimg * config['ema_rampup'])
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= config['total_kimg'] * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + 4 * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        print(' '.join(fields))


        # Save image snapshot.
        if (config['image_snapshot_ticks'] is not None) and (done or cur_tick % config['image_snapshot_ticks'] == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            util.save_image_grid(images, os.path.join(config['out_dir'], f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)


        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (config['network_snapshot_ticks'] is not None) and (done or cur_tick % config['network_snapshot_ticks'] == 0):
            snapshot_data = dict(training_set_kwargs=dict(config))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(config['out_dir'], f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        metrics = ['fid50k_full']
        if (snapshot_data is not None) and (len(metrics) > 0):
            print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=config, num_gpus=1, rank=0, device=device)
                metric_main.report_metric(result_dict, run_dir=config['out_dir'], snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory


        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    print()
    print('Exiting...')
