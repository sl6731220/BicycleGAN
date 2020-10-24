"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from IPython import embed

#数据搜索
def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    #dataset_name = 'aligned'，其实这一句就相当于生成了目录'data.aligned_dataset'
    dataset_filename = "data." + dataset_name + "_dataset"
    #动态导入模块'data.aligned_dataset(一个py文件)，动态导入模块允许我们通过‘字符串形’式来导入模块，写起来更方便
    #AlignedDataset这个类才是关键
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    #'aligneddataset'  str.replace(old, new[, max]) 
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    #万物皆对象，取出这个模块的所有属性
    #这一部分主要用于判断
    for name, cls in datasetlib.__dict__.items():
        #issubclass(class, classinfo)用于判断参数 class 是否是类型参数 classinfo 的子类
        #lower() 方法转换字符串中所有大写字符为小写
        #找到'aligneddataset'这一项是否在datasetlib中存在，并且是否符合BaseDataset这个模板的格式，是他的子类
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
           #data.aligned_dataset这个模块下面只有AlignedDataset这个类所以取出来的cls就是这个类
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        创建数据集实列和多线程加载器
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        #进行数据搜索
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        #加载的数据集
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
