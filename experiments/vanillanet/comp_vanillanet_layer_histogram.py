import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import models.vanillanet
from timm.models import create_model
from comp_full_gradient import str2bool

def parse_args():
    parser = argparse.ArgumentParser(
        description='For Vanillanet, compute the histogram of in_ch*out_ch.'
    )
    parser.add_argument(
        '--drop', 
        type=float, 
        default=0, 
        metavar='PCT',
        help='Drop rate (default: 0.0)'
    )
    parser.add_argument(
        '--act_num', 
        default=3, 
        type=int
    )
    parser.add_argument(
        '--nb_classes', 
        default=1000, 
        type=int,
        help='number of the classification types'
    )
    parser.add_argument(
        '--deploy', 
        type=str2bool, 
        default=False
    )
    args = parser.parse_args()

    return args


def get_conv_cin_cout_products(model, use_effective_cin=True):
    """
    Returns list of C_in * C_out for every Conv2d layer.

    use_effective_cin=True means grouped conv uses C_in/groups.
    """
    vals = []
    layer_info = []

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            cin = m.in_channels
            cout = m.out_channels
            groups = m.groups

            effective_cin = cin // groups if use_effective_cin else cin
            prod = effective_cin * cout

            vals.append(prod)
            layer_info.append({
                "name": name,
                "cin": cin,
                "cout": cout,
                "groups": groups,
                "effective_cin": effective_cin,
                "cin_x_cout": prod,
                "kernel": m.kernel_size,
            })

    return vals, layer_info


def plot_cin_cout_hist(
    model, 
    save_path=None
):
    """
    models_dict: {"UNet": unet_model, "Vanillanet": sq_model, ...}
    """
    plt.figure(figsize=(8, 5))

    vals, _ = get_conv_cin_cout_products(model)
    plt.hist(
        vals, 
        bins='auto', 
        alpha=0.45
    )

    plt.xlabel(r"$C_{in} \times C_{out}$")
    plt.ylabel("Number of Conv2d layers")
    plt.title(r"Vanillanet histogram of $C_{in} \times C_{out}$ across Conv2d layers")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    print("Saving in {}".format(save_path))
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vanillanet_10"

    model = create_model(
            model_name, 
            pretrained=False,
            num_classes=args.nb_classes, 
            act_num=args.act_num,
            drop_rate=args.drop,
            deploy=args.deploy,
    )

    model = model.to(device)

    plot_cin_cout_hist(
        model = model,
        save_path="vanillanet_cin_cout_histogram.png",
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)