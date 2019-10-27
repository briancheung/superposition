import pprint
import os
import socket
import json
import time
from datetime import datetime
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as tvu 
from tensorboardX import SummaryWriter

import datasets as ds 
from configs import paramsuper, getters


def get_model_params(model, clone=True):
    myparams = []
    for parameter in model.parameters():
        if clone:
            myparams.append(parameter.clone())
        else:
            myparams.append(parameter)

    return myparams


def get_uncoupled_norm(x_list, y_list):
    sqsum = 0.
    for x,y in zip(x_list, y_list):
       sqsum += ((x - y)**2).sum().item()
    norm = np.sqrt(sqsum)
    return norm


def train(model, optimizer, time, data, loss_coeffs):
    input_data, target_data = data
    time_loss_coeff, s_loss_coeff = loss_coeffs

    optimizer.zero_grad()
    out_a, out_b, preacts = model(input_data, time)
    out_class = out_a
    logsm_class = F.log_softmax(out_class, 1)
    sm_class = F.softmax(out_class, 1)

    # Calculate entropy of classes
    loss_entropy = (-sm_class*logsm_class).sum(1).mean()
    loss_class = F.nll_loss(logsm_class, target_data)

    loss = loss_class
    loss.backward()
    optimizer.step()
    return loss, loss_class, loss_entropy


def test_set(model, test_loader, device, time, period, preprocess, steps):
    test_loss_class = 0
    test_loss_time = 0
    test_loss_entropy = 0
    correct = 0
    num_seen = 0
    model.eval()
    with torch.no_grad():
        for batch_idx in range(steps):
            # Set time before getting data to get correct angle
            test_loader.set_time(time*period - 1)
            input_data, target = test_loader.get_data()
            input_data, target = input_data.to(device), target.to(device)
            pp_input = preprocess(input_data)
            out_a, out_b, preacts = model(pp_input, time)

            out_class = out_a
            logsm_class = F.log_softmax(out_class, 1)
            sm_class = F.softmax(out_class, 1)

            loss_entropy = (-sm_class*logsm_class).sum()
            loss_class = F.nll_loss(logsm_class, target, reduction='sum')

            test_loss_class += loss_class.item()
            test_loss_entropy += loss_entropy.item()
            pred = logsm_class.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()
            num_seen += input_data.shape[0]

    test_loss_class /= num_seen
    test_loss_entropy /= num_seen
    test_acc_class = 100. * correct / num_seen 
    print('\nTest set: Time: {:5f}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        time, test_loss_class, correct, num_seen,
        test_acc_class))

    return test_loss_class, test_acc_class, test_loss_entropy


def main(args):
    pprint.pprint(vars(args))

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = getters.get_dataset(args.dataset, args.period, args.batch_size, True, kwargs)
    test_loader = getters.get_dataset(args.dataset, args.period, args.test_batch_size, False, kwargs)
    activation = getters.get_activation(args.activation)
    input_dim, output_dim = train_loader.get_dim()
    mynet = getters.get_fc_net(args.net, input_dim, output_dim, activation, args)  
    if mynet:
        flat_input = True
    else:
        mynet = getters.get_conv_net(args.net, input_dim, output_dim, activation, args)
        flat_input = False 
    mynet = mynet.to(device)
    optimizer = getters.get_optimizer(args.optimizer, mynet.parameters(), args)

    def get_preprocess(flatten=False):
        if flatten:
            return lambda x: x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        return lambda x: x
    preprocess = get_preprocess(flat_input)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs',
                           args.desc,
                           current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Args', pprint.pformat(vars(args)), 0)
    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=4)

    for batch_idx in range(args.steps):
        global_step = batch_idx
        if global_step < args.stationary:
            train_loader.set_time(np.random.randint(args.period))

        time_start = time.time()
        input_data, target = train_loader.get_data()
        input_data, target = input_data.to(device), target.to(device)
        pp_input = preprocess(input_data)

        mynet.train()
        if batch_idx % 100 == 0:
            params_tm1 = get_model_params(mynet, clone=True)
        net_time = train_loader.time() % args.cheat_period
        net_time /= args.time_slow
        losses = train(mynet,
                       optimizer,
                       net_time,
                       (pp_input, target),
                       (args.time_loss_coeff, args.s_loss_coeff))
        if batch_idx % 100 == 0:
            params_t = get_model_params(mynet, clone=False)
            norm_delta_params = get_uncoupled_norm(params_tm1, params_t)

        loss, loss_class, loss_entropy = losses
        time_stop = time.time()

        if batch_idx % 100 == 0:
            print(batch_idx, loss.item(), train_loader.current_time, time_stop-time_start)
            if args.shuffle_test:
                test_time = np.random.randint(args.period)
            else:
                test_time = args.test_time

            test_time = test_time % args.cheat_period
            test_time /= args.time_slow
            test_losses = test_set(mynet,
                                   test_loader,
                                   device,
                                   test_time,
                                   args.period,
                                   preprocess,
                                   args.test_steps)
            test_loss_class, test_acc_class, test_loss_entropy = test_losses

            writer.add_scalar('norm_delta_params', norm_delta_params, global_step)
            writer.add_scalar('local_loss_class', loss_class, global_step)
            writer.add_scalar('local_loss_entropy', loss_entropy, global_step)
            writer.add_scalar('local_train_loss', loss, global_step)
            writer.add_scalar('test_acc_class', test_acc_class, global_step)
            writer.add_scalar('test_loss_class', test_loss_class, global_step)
            writer.add_scalar('test_loss_entropy', test_loss_entropy, global_step)
            writer.add_scalar('system_loop_time', time_stop-time_start, global_step) 
            img = tvu.make_grid(input_data[:32], normalize=True)
            writer.add_image('train_image', img, global_step)

        if batch_idx % 1000 == 0:
            save_path = os.path.join(log_dir, 'mynet_%d.pth' % batch_idx)
            torch.save(mynet, save_path)

    # Test on all the tasks to get an average accuracy accross all tasks
    n_tasks = int(args.steps/args.period)
    total_acc = 0.
    for task_i in range(n_tasks):
        test_time = task_i 
        test_losses = test_set(mynet,
                               test_loader,
                               device,
                               test_time,
                               args.period,
                               preprocess,
                               args.test_steps)
        test_loss_class, test_acc_class, test_loss_entropy = test_losses
        writer.add_scalar('retro_acc', test_acc_class, task_i)
        print('test_time:', test_time, 'acc:', test_acc_class)
        total_acc += test_acc_class

    writer.add_scalar('avg_acc', total_acc/n_tasks, global_step)
    writer.close()


if __name__ == "__main__":
    rotmnist_exps = [paramsuper.RotatingMNISTUnitaryHash(),
                     paramsuper.RotatingMNISTUnitaryNLKHash(),
                     paramsuper.RotatingMNISTPytorch(),
                     paramsuper.RotatingMNISTComplex(),
                     paramsuper.RotatingMNISTReal()]

    permmnist_units_exps = [paramsuper.PermutingMNISTBinaryHash128(),
                            paramsuper.PermutingMNISTBinaryHash256(),
                            paramsuper.PermutingMNISTBinaryHash512(),
                            paramsuper.PermutingMNISTBinaryHash1024(),
                            paramsuper.PermutingMNISTBinaryHash2048()]

    permmnist_alg_exps = [paramsuper.PermutingMNISTBinaryHash(),
                          paramsuper.PermutingMNISTPytorch()]

    fmnist_exps = [paramsuper.RotatingFMNISTBinary(),
                   paramsuper.RotatingFMNISTBinary10L()]

    icifar_exps = [paramsuper.ICIFARResNet18(),
                   paramsuper.ICIFARMultiResNet18(),
                   paramsuper.ICIFARHashResNet18(),
                   paramsuper.ICIFAR100ResNet18(),
                   paramsuper.ICIFAR100HashResNet18()]

    exps_to_run = (rotmnist_exps + 
                   permmnist_units_exps +
                   permmnist_alg_exps + 
                   fmnist_exps +
                   icifar_exps)

    for args in exps_to_run:
        main(args)
