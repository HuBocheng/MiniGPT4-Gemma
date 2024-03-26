# MiniGPT4-Gemma

Optimization of the LLM backbone network in MINIGPTV, replacing the original llama with gemma.



# AIM

gemma去替换miniGPT4V中的llama，然后训练一个多模态版本

仓库：https://github.com/Vision-CAIR/MiniGPT-4；



# 骨干网优化项目整体思路

1. 用原有第一阶段的训练方式做文本和图片端的对齐，让模型可以识别图片并且可以基本完成一些视觉任务--->证明it works:white_check_mark:
2. 用特定分领域内的数据集做微调，需要构建相应的指令模板，让其适配其他specific的任务
3. 尝试加入视频理解相关能力



# 完成进度

:white_check_mark:【2024/3/26】llm backbone由llama替换成gemma（gemma-2b-it），CC3M+CC12M+SBU数据集训练6epoch（20000次迭代），模型基本对齐了视觉端和语言端，可以回答一些基本图像问题。框架同时**支持llama和gemma模型的chat和训练**



# 项目复现：

### Chat

1. eval配置文件填入训练好的模型检查点权重路径：eval_configs/minigpt4_gemma_eval.yaml
2. 填入要使用的llm_backbone（llama、gemma）的路径：minigpt4/configs/models/minigpt4_llama2.yaml或者minigpt4/configs/models/minigpt4_gemma.yaml
3. 根目录下创建hf_token.txt并填入自己账户的hf_token，使用gemma的接口需要再huggingface上申请token，链接：[here](https://huggingface.co/settings/tokens)
4. 执行demo.py：`python demo.py --cfg-path eval_configs/minigpt4_llama2_eval.yaml  --gpu-id 0` 或者 python demo.py --cfg-path eval_configs/minigpt4_gemma_eval.yaml  --gpu-id 0

### 训练

1. 在chat设置的基础上设置数据集路径：minigpt4/configs/datasets/laion/defaults.yaml、minigpt4/configs/models/minigpt4_llama2.yaml
2. 可以适当修改train_configs/minigpt4_gemma_stage1_pretrain.yaml中的训练参数
3. 执行命令：`python train.py --cfg-path train_configs/minigpt4_gemma_stage1_pretrain.yaml`



# START

mac性能好于win，想在mac上部署minigptv，但是mac不支持CUDA，其mrs拓展支持mac使用pytorch但是现有部分库还是不行

换了win本，解决了一些报错，遇到的报错仓库的issue里都有，有一个需要注意的是用本身的environment.yml生成conda虚拟环境时候自动下载的是torch 2.0的CPU版本，需要换成GPU版本

适当修改了代码的两个地方，有些地方win不兼容。

demo_v2.py：

```Python
def save_tmp_img(visual_img):
    file_name = "".join([str(random.randint(0, 9)) for _ in range(5)]) + ".jpg"
    file_path = "tmp/gradio" + file_name  # 修改
    visual_img.save(file_path)
    return file_path
```

eva_vit.py：

```Python
url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    #  cached_file = download_cached_file(
    #     url, check_hash=False, progress=True
    # )
    print("无需下载，使用本地的eva_vit_g.pth")
    local_saved_vit_g_filepath = "weight/eva_vit_g.pth"
    cached_file = local_saved_vit_g_filepath
    state_dict = torch.load(cached_file, map_location="cpu")
    interpolate_pos_embed(model, state_dict)
```



# stage0：整体理解

### 注册表Registy机制

- 构建器：数据集构建器
- 任务：image_text_pretrain
- 处理器：blip_caption、blip_train、blip_eval
- 模型：minigpt4、minigpt4v
- 运行器：base
- 学习率调整：'linear_warmup_step_lr'、'linear_warmup_cosine_lr'

### bulider文件：执行器，用于执行/创建一些组件

1. **BaseDatasetBuilder**
   1. 属性有train_dataset_cls、train_dataset_cls、self.config、self.data_type、**self.vis_processors、self.text_processors**
      1. self.config默认是None，类的构造方法指定self.config是None的时候就去指定位置（DATASET_CONFIG_DICT[type]，type默认是'default'）load一个cfg
   2. `build_datasets`方法：在子类没有重写该方法的时候调用，用于下载数据集并且执行**子类数据集构建器**/父类本身的bulid方法。本质是调用`build()`方法得到一个名为dataset的字典返回（包含dataset的类型【train/eval】和一个数据集load对象【类似pytorch原生的dataloader，可以是自定义的一个继承于BaseDataset的类、也可是`webdataset`库的DataPipeline对象——方便处理tar格式组织的数据集】）
   3. `build_processors`方法：为当前的数据集构建器类添加处理器，总之是在类中添加两个字典`vis_processors`、和`text_processors`，保存了视觉数据和文本数据分别要使用的处理器类
   4. `build方法`方法：调用`build_processors`方法--->
   5. 
2. 众多**BaseDatasetBuilder的子类**（在image_text_pair_builder.py中）
   1. 属性有`DATASET_CONFIG_DICT`，保存着这个数据集的说明信息（yaml格式储存，统一命名为default.yaml，不同数据集的保存在不同文件夹）
   2. `build_datasets`方法：可以重写可以调用父类的
   3. 

### models文件夹：整个网络的结构（有骨干网络有组件）

1. BaseModel继承nn.Module，有子类MiniGPTBase
2. MiniGPTBase有子类MiniGPTv2、MiniGPT4
   1. 属性指定了VIT模型名，imgsize之类的东西，这些在MiniGPTv2、MiniGPT4相同的，简化MiniGPTv2、MiniGPT4代码
3. 组件类
4. MiniGPT4
   1. 属性有PRETRAINED_MODEL_CONFIG_DICT，指出用的两个LLM的配置.yaml文件的位置
5. 总之：总bese下初始化LLM；minigptbase下初始化视觉编码器；gpt4下初始化 Q-Former



### processors文件夹：处理器

这些处理器都在构建器中实例化并在在之后处理任务



### 封装组织与类划分

**task**：Basetask下有ImageTextPretrainTask子类

**builder**：BaseDatasetBuilder下有所有数据集子类的构建器，有的子类重写父类`build_datasets`函数

**dataset**：BaseDataset下有所有数据集的对象，这些对象由对应的数据集构建器构建，内部还用`webdataset`库创建内部数据集pipeline对象（定义里数据集位置，对tar文件的操作，随机打乱，decode方法，数据集组织方法等等），

**model**：BaseModel、Minigptmodel、



### 整体流程

```
tasks.setup_task(cfg)`在cfg和注册表中找到task的对应的类并实例化一个task对象--->`task.bulid_datasets`获得完整数据集对象（类似Dataloader）--->`task.build_model(cfg)
```

`task.bulid_datasets`遍历yml中涉及的数据集，找到各自对应的数据集构建器类构建数据集对象dataset，这个dataset是一个字典（包含dataset的类型【train/eval】和一个数据集load对象【类似pytorch原生的dataloader，可以是自定义的一个继承于BaseDataset的类、也可是`webdataset`库的DataPipeline对象——方便处理tar格式组织的数据集】）

- 用数据集对应的构建器中的`build_datasets`方法构建数据集对象得到dataset字典
  - 判断要不要下载数据集、要不要开启分布式训练环境
  - `build_datasets`接受task任务对应子类的`build`函数返回值并返回
    - `self.build`先调用`build_processors`给数据集类中添加处理器，添加两个两个字典`vis_processors`、和`text_processors`（字典的键是train/eval，值是对应的`processor`类）
    - 找到要加载数据集的位置等信息build_info
    - 用相应的配置实例化数据集加载器XXXDataset（不是XXXDatasetBulider了）叫做dataset并返回
- 给dataset字典附上数据集名称name_i和抽样比率sample_ratio

```
task.build_model
```

- 首先读取cfg提取model部分的cfg，找到对应的model类（minigpy4/minigptv）



# stageX：代码优化

~~1、gemma的硬编码和hf_token的硬编码，对LLM骨架类型的指定应该放在model的yaml里还是task的yaml里~~





# 问题及解决

**AWS**没有GPU和CUDA：需要换image（用G5-Xlarge的镜像），利用conda虚拟环境，代码一些部分需要做适配性修改，尽量避免云端下载数据，权重等数据需要提前下载缓存。

**laion数据集**的索引json文件过大炸内存*：改用ijson逐项处理JSON数据，避免直接载入整个JSON进内存

**数据集下载渠道复杂**：写了一个简单脚本一键下载：用wget逐项下载Minigptv2训练所需的数据集

**数据集加载器**：`DataPipeline`在处理大规模、文件存储的数据集时性能更优，可以直接从压缩文件中流式读取数据，减少了IO开销

PyTorch的`DataLoader`是一个通用的数据加载工具，适用于各种数据集，特别是那些已经完全加载到内存或需要逐个文件访问的数据集。

**大模型预训练阶段的数据集下载**：img2dataset工具在云端使用还有一些问题，链接总是断开，有数据损坏的情况出现，为了避免潜在风险，开了新的镜像和conda环境专门用于数据集下载，不报错了但是速度感人，差不多20shards/h，中间还有断连。（小数据集共1200+shards）



<span style="background-color: #d0d0d0; color: black;">这是有灰色背景的文本</span>

**环境配置与代码问题**

<span style="background-color: #d0d0d0; color: black;">ImportError: libGL.so.1: cannot open shared object file: No such file or directory</span>

`pip install opencv-python-headless`



<span style="background-color: #d0d0d0; color: black;">TypeError: forward() got an unexpected keyword argument</span>

不传相应参数，代码换种写法就行



<span style="background-color: #d0d0d0; color: black;">没有accelerate包</span>minigpt-4中的accelerate包貌似不支持win，用下面这个好解决

`pip install git+https://github.com/huggingface/accelerate`



<span style="background-color: #d0d0d0; color: black;">UserWarning: Using the update method is deprecated. Simply return a new object instead,
e.g. return gr.Textbox(...) instead of return gr.Textbox.update(...)</span>
需要降低版本 gradio==3.39.0



<span style="background-color: #d0d0d0; color: black;">TypeError: LlavaLlamaForCausalLM.forward() got an unexpected keyword argument 'cache_position'</span>
确保 transformers 版本与 pyproject.toml 中提到的版本相同来修复它



<span style="background-color: #d0d0d0; color: black;">命令行下载Google Drive上的文件</span>
pip install gdown
原链接：https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view?usp=sharing

从共享链接中提取文件ID并使用gdown工具:gdown 'https://drive.google.com/uc?id=11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk'



