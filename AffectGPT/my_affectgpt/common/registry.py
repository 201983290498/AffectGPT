"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


class Registry:
    mapping = {
        "builder_name_mapping": {},
        "task_name_mapping": {},
        "processor_name_mapping": {},
        "model_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "runner_name_mapping": {},
        "visual_encoder_mapping": {},
        "acoustic_encoder_mapping": {},
        "state": {},
        "paths": {},
    }

    @classmethod
    def register_builder(cls, name):
        r"""将数据集构建器以键 'name' 注册到注册表

        参数:
            name: 用于注册构建器的键。

        用法:

            from affectgpt.common.registry import registry
            from affectgpt.datasets.base_dataset_builder import BaseDatasetBuilder
        """

        def wrap(builder_cls):
            from my_affectgpt.datasets.builders.base_dataset_builder import BaseDatasetBuilder

            assert issubclass(
                builder_cls, BaseDatasetBuilder
            ), "所有构建器必须继承 BaseDatasetBuilder 类, 当前为 {}".format(
                builder_cls
            )
            if name in cls.mapping["builder_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["builder_name_mapping"][name]
                    )
                )
            cls.mapping["builder_name_mapping"][name] = builder_cls
            return builder_cls

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""将任务以键 'name' 注册到注册表

        参数:
            name: 用于注册任务的键。

        用法:

            from affectgpt.common.registry import registry
        """

        def wrap(task_cls):
            from my_affectgpt.tasks.base_task import BaseTask

            assert issubclass(
                task_cls, BaseTask
            ), "所有任务必须继承 BaseTask 类"
            if name in cls.mapping["task_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["task_name_mapping"][name]
                    )
                )
            cls.mapping["task_name_mapping"][name] = task_cls
            return task_cls

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""将模型以键 'name' 注册到注册表

        参数:
            name: 用于注册模型的键。

        用法:

            from affectgpt.common.registry import registry
        """

        def wrap(model_cls):
            from my_affectgpt.models import BaseModel

            assert issubclass(
                model_cls, BaseModel
            ), "所有模型必须继承 BaseModel 类"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return wrap

    @classmethod
    def register_processor(cls, name):
        r"""将处理器以键 'name' 注册到注册表

        参数:
            name: 用于注册处理器的键。

        用法:

            from affectgpt.common.registry import registry
        """

        def wrap(processor_cls):
            from my_affectgpt.processors import BaseProcessor

            assert issubclass(
                processor_cls, BaseProcessor
            ), "所有处理器必须继承 BaseProcessor 类"
            if name in cls.mapping["processor_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["processor_name_mapping"][name]
                    )
                )
            cls.mapping["processor_name_mapping"][name] = processor_cls
            return processor_cls

        return wrap


    @classmethod
    def register_visual_encoder(cls, name):
        # 注册视觉编码器
        def wrap(encoder_cls):
            if name in cls.mapping["visual_encoder_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["visual_encoder_mapping"][name]
                    )
                )
            cls.mapping["visual_encoder_mapping"][name] = encoder_cls
            return encoder_cls

        return wrap


    @classmethod
    def register_acoustic_encoder(cls, name):
        # 注册声学编码器
        def wrap(encoder_cls):
            if name in cls.mapping["acoustic_encoder_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["acoustic_encoder_mapping"][name]
                    )
                )
            cls.mapping["acoustic_encoder_mapping"][name] = encoder_cls
            return encoder_cls

        return wrap


    @classmethod
    def register_lr_scheduler(cls, name):
        r"""将学习率调度器以键 'name' 注册到注册表

        参数:
            name: 用于注册调度器的键。

        用法:

            from affectgpt.common.registry import registry
        """

        def wrap(lr_sched_cls):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_sched_cls
            return lr_sched_cls

        return wrap

    @classmethod
    def register_runner(cls, name):
        r"""将运行器以键 'name' 注册到注册表

        参数:
            name: 用于注册运行器的键。

        用法:

            from affectgpt.common.registry import registry
        """

        def wrap(runner_cls):
            if name in cls.mapping["runner_name_mapping"]:
                raise KeyError(
                    "名称 '{}' 已注册为 {}。".format(
                        name, cls.mapping["runner_name_mapping"][name]
                    )
                )
            cls.mapping["runner_name_mapping"][name] = runner_cls
            return runner_cls

        return wrap

    @classmethod
    def register_path(cls, name, path):
        r"""将路径以键 'name' 注册到注册表

        参数:
            name: 用于注册路径的键。

        用法:

            from affectgpt.common.registry import registry
        """
        assert isinstance(path, str), "所有路径必须为字符串。"
        if name in cls.mapping["paths"]:
            raise KeyError("名称 '{}' 已注册。".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def register(cls, name, obj):
        r"""将项以键 'name' 注册到注册表

        参数:
            name: 用于注册项的键。

        用法::

            from affectgpt.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".") # ['configuration']
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj # 添加 'configuration' -> <affectgpt.common.config.Config object at 0x7f9bf84b1370>


    '''
    {'cc_sbu': <class 'affectgpt.datasets.builders.image_text_pair_builder.CCSBUBuilder'>, 
    'laion': <class 'affectgpt.datasets.builders.image_text_pair_builder.LaionBuilder'>, 
    'cc_sbu_align': <class 'affectgpt.datasets.builders.image_text_pair_builder.CCSBUAlignBuilder'>, 
    'webvid': <class 'affectgpt.datasets.builders.video_caption_builder.WebvidBuilder'>, 
    'instruct': <class 'affectgpt.datasets.builders.instruct_builder.Instruct_Builder'>, 
    'webvid_instruct': <class 'affectgpt.datasets.builders.instruct_builder.WebvidInstruct_Builder'>, 
    'webvid_instruct_zh': <class 'affectgpt.datasets.builders.instruct_builder.WebvidInstruct_zh_Builder'>, 
    'llava_instruct': <class 'affectgpt.datasets.builders.instruct_builder.LlavaInstruct_Builder'>}
    '''
    @classmethod
    def get_builder_class(cls, name):
        return cls.mapping["builder_name_mapping"].get(name, None)
    
    @classmethod
    def get_visual_encoder_class(cls, name):
        return cls.mapping["visual_encoder_mapping"].get(name, None)
    
    @classmethod
    def get_acoustic_encoder_class(cls, name):
        return cls.mapping["acoustic_encoder_mapping"].get(name, None)

    '''
    {'affectgpt': <class 'affectgpt.models.affectgpt.Af'>}
    '''
    @classmethod
    def get_model_class(cls, name): # name = 'affectgpt'
        return cls.mapping["model_name_mapping"].get(name, None) # get是一种从map中读取数据的方式，更加不容易报错的方法

    '''
    {'image_text_pretrain': <class 'affectgpt.tasks.image_text_pretrain.ImageTextPretrainTask'>, 
    'video_text_pretrain': <class 'affectgpt.tasks.video_text_pretrain.VideoTextPretrainTask'>}
    '''
    @classmethod
    def get_task_class(cls, name):
        return cls.mapping["task_name_mapping"].get(name, None)

    @classmethod
    def get_processor_class(cls, name):
        return cls.mapping["processor_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_class(cls, name):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_runner_class(cls, name):
        return cls.mapping["runner_name_mapping"].get(name, None)

    '''
    {'runner_base': <class 'affectgpt.runners.runner_base.RunnerBase'>}
    '''
    @classmethod
    def list_runners(cls):
        return sorted(cls.mapping["runner_name_mapping"].keys())

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def list_tasks(cls):
        return sorted(cls.mapping["task_name_mapping"].keys())

    @classmethod
    def list_processors(cls):
        return sorted(cls.mapping["processor_name_mapping"].keys())

    @classmethod
    def list_lr_schedulers(cls):
        return sorted(cls.mapping["lr_scheduler_name_mapping"].keys())

    @classmethod
    def list_datasets(cls):
        return sorted(cls.mapping["builder_name_mapping"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""从注册表中以键 'name' 获取项

        参数:
            name (string): 需要检索值的键。
            default: 如果未找到键且传入该参数，则返回默认值并警告。默认: None
            no_warning (bool): 如果为 True，则未找到键时不警告。用于 MMF 的内部操作。默认: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "键 {} 不在注册表中, 返回默认值 {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""从注册表中移除以键 'name' 的项

        参数:
            name: 需要移除的键。
        用法::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
