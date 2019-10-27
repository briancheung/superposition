class RotatingMNISTBinaryHash:
    def __init__(self):
        self.activation = 'relu'
        self.batch_size = 128
        self.cheat_period = 1000
        self.dataset = 'rotating_mnist'
        self.desc = 'rot_mnist_binhash'
        self.key_pick = 'hash'
        self.learn_key = True
        self.lr = 1e-4
        self.momentum = 0.5
        self.n_layers = 3
        self.n_units = 256
        self.net = 'binaryhash'
        self.net_period = 50
        self.no_cuda = False
        self.optimizer = 'rmsprop'
        self.period = 1000
        self.rotate_continually = False
        self.seed = 1
        self.shuffle_test = False
        self.stationary = 0
        self.steps = 50000
        self.test_batch_size = 1000
        self.test_steps = 10
        self.test_time = 0
        self.time_slow = 100.
        self.time_loss_coeff = 0.
        self.s_loss_coeff = 0.


class RotatingMNISTComplexHash(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTComplexHash, self).__init__()
        self.net = 'hash'
        self.desc = 'rot_mnist_hash'


class RotatingMNISTUnitaryHash(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTUnitaryHash, self).__init__()
        self.net = 'rotatehash'
        self.desc = 'rot_mnist_rothash'


class RotatingMNISTUnitaryNLKHash(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTUnitaryNLKHash, self).__init__()
        self.net = 'rotatehash'
        self.desc = 'rot_mnist_rothash'
        self.learn_key = False


class RotatingMNISTPytorch(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTPytorch, self).__init__()
        self.net = 'pytorch'
        self.desc = 'rot_mnist_pytorch'


class RotatingMNISTComplex(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTComplex, self).__init__()
        self.net = 'complex'
        self.desc = 'rot_mnist_complex'


class RotatingMNISTReal(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingMNISTReal, self).__init__()
        self.net = 'real'
        self.desc = 'rot_mnist_real'


class RotatingFMNISTBinary(RotatingMNISTBinaryHash):
    def __init__(self):
        super(RotatingFMNISTBinary, self).__init__()
        self.dataset = 'rotating_fashionmnist'
        self.desc = 'rot_fmnist_binhash'


class RotatingFMNISTBinary10L(RotatingFMNISTBinary):
    def __init__(self):
        super(RotatingFMNISTBinary10L, self).__init__()
        self.n_layers = 10
        self.desc = 'rot_fmnist_binhash10l'


class PermutingMNISTBinaryHash:
    def __init__(self):
        self.activation = 'relu'
        self.batch_size = 128
        self.cheat_period = 1000000
        self.dataset = 'permuting_mnist'
        self.desc = 'permuting_mnist_marathon_binhash'
        self.key_pick = 'hash'
        self.learn_key = True
        self.lr = 1e-4
        self.momentum = 0.5
        self.n_layers = 3
        self.n_units = 256
        self.net = 'binaryhash'
        self.net_period = 50
        self.no_cuda = False
        self.optimizer = 'rmsprop'
        self.period = 1000
        self.rotate_continually = False
        self.seed = 1
        self.shuffle_test = False
        self.stationary = 0
        self.steps = 50000
        self.test_batch_size = 1000
        self.test_steps = 10
        self.test_time = 0
        self.time_slow = 1000.
        self.time_loss_coeff = 0.
        self.s_loss_coeff = 0.


class PermutingMNISTBinaryHash128(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTBinaryHash128, self).__init__()
        self.n_units = 128
        self.desc = 'permuting_mnist_marathon_binhash128'


class PermutingMNISTBinaryHash256(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTBinaryHash256, self).__init__()
        self.n_units = 256 
        self.desc = 'permuting_mnist_marathon_binhash256'


class PermutingMNISTBinaryHash512(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTBinaryHash512, self).__init__()
        self.n_units = 512 
        self.desc = 'permuting_mnist_marathon_binhash512'


class PermutingMNISTBinaryHash1024(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTBinaryHash1024, self).__init__()
        self.n_units = 1024 
        self.desc = 'permuting_mnist_marathon_binhash1024'


class PermutingMNISTBinaryHash2048(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTBinaryHash2048, self).__init__()
        self.n_units = 2048 
        self.desc = 'permuting_mnist_marathon_binhash2048'


class PermutingMNISTPytorch(PermutingMNISTBinaryHash):
    def __init__(self):
        super(PermutingMNISTPytorch, self).__init__()
        self.net = 'pytorch'
        self.desc = 'permuting_mnist_marathon_pytorch'


class ICIFARResNet18:
    def __init__(self):
        self.activation = 'relu'
        self.batch_size = 128
        self.cheat_period = 100000
        self.dataset = 'incrementing_cifar'
        self.desc = 'icifar_resnet18'
        self.key_pick = 'hash'
        self.learn_key = True
        self.lr = 0.001
        self.momentum = 0.5
        self.n_layers = 6
        self.n_units = 64
        self.net = 'staticbnresnet18'
        self.net_period = 10
        self.no_cuda = False
        self.optimizer = 'rmsprop'
        self.period = 20000
        self.rotate_continually = False
        self.seed = 1
        self.shuffle_test = False
        self.stationary = 0
        self.steps = 100000
        self.test_batch_size = 1000
        self.test_steps = 10
        self.test_time = 0
        self.time_slow = 20000.
        self.time_loss_coeff = 0.
        self.s_loss_coeff = 0.
        

class ICIFAR100ResNet18(ICIFARResNet18):
    def __init__(self):
        super(ICIFAR100ResNet18, self).__init__()
        self.dataset = 'incrementing_cifar100'
        self.desc = 'icifar100_resnet18'


class ICIFARMultiResNet18(ICIFARResNet18):
    def __init__(self):
        super(ICIFARMultiResNet18, self).__init__()
        self.net = 'multiresnet18'
        self.desc = 'icifar_multiresnet18'


class ICIFARHashResNet18(ICIFARResNet18):
    def __init__(self):
        super(ICIFARHashResNet18, self).__init__()
        self.net = 'hashresnet18'
        self.desc = 'icifar_hashresnet18'


class ICIFAR100HashResNet18(ICIFAR100ResNet18):
    def __init__(self):
        super(ICIFAR100HashResNet18, self).__init__()
        self.net = 'hashresnet18'
        self.desc = 'icifar100_hashresnet18'

