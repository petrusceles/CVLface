from ..base import BaseModel
from .model import IR_50, IR_18
from torchvision import transforms


class IResNetModelDsc(BaseModel):
    """
    A class representing a model for IResNet architectures. It supports creating
    models with specific configurations such as IR_50 and IR_101.

    Attributes:
        net (torch.nn.Module): The IResNet network (either IR_50 or IR_101).
        config (object): The configuration object with model specifications.
    """

    def __init__(self, net, config):
        super(IResNetModelDsc, self).__init__(config)
        self.net = net
        self.config = config

    @classmethod
    def from_config(cls, config):
        if config.name == "ir50dsc":
            net = IR_50(input_size=(112, 112), output_dim=config.output_dim)
        elif config.name == "ir18dsc":
            net = IR_18(input_size=(112, 112), output_dim=config.output_dim)
        else:
            raise NotImplementedError

        model = cls(net, config)
        model.eval()
        return model

    def forward(self, x):
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x)

    def make_train_transform(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform

    def make_test_transform(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform


def load_model(model_config):
    model = IResNetModelDsc.from_config(model_config)
    return model
