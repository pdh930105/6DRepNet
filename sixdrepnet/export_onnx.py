import torch
from easydict import EasyDict
import numpy as np
from pathlib import Path
import argparse
from model import SixDofNet, SixDRepNet
from convert import repvgg_model_convert

def file_size(file):
    # Return file size in MB
    return Path(file).stat().st_size / 1e6

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='snapshot path')
    parser.add_argument('--output_onnx', type=str, default='', help='output name')
    parser.add_argument('--img-size', type=int, default=224, help='image size (default=224)')
    parser.add_argument('--dynamic', action='store_true', help='using dynamic batch size')
    parser.add_argument('--simplify', action='store_true', help='using onnx-simplify')

    opt = parser.parse_args()
    prefix = 'ONNX:'

    img = torch.zeros([1,3,opt.img_size, opt.img_size], dtype=torch.float32)

    model_path = Path(opt.weights)
    assert model_path.exists(), f"snapshot {opt.weights} does not exist"

    model_name = model_path.parts[-2].split('_')[0]
    model_train_img_size = int(model_path.parts[-2].split('_')[-1])
    assert model_train_img_size == opt.img_size, f"does not same train model img size {model_train_img_size} != {opt.img_size}"
    output_names = ['output']
    print("model name : ", model_name)
    if opt.output_onnx == '':
        opt.output_onnx = f'{model_name}_dof.onnx'

    model_checkpoint = torch.load(model_path)

    if 'RepVGG' in model_name:
        model = SixDRepNet(backbone_name=model_name,
                                    backbone_file='',
                                    deploy=False,
                                    pretrained=False)
        print('load model :', model_path)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        print('Converting RepVGG model for inference')
        model = repvgg_model_convert(model, save_path=None)
        print('Done.')

    else:
        print('load model :', model_path)
        model = SixDofNet(model_name, pretrained=False)
        model.load_state_dict(model_checkpoint['model_state_dict'])
        print('Done.')

    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        
        torch.onnx.export(model, img, opt.output_onnx, verbose=False, opset_version=11, input_names=['images'], output_names=output_names,
                                dynamic_axes=None)
        # Checks
        model_onnx = onnx.load(opt.output_onnx)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print
        # Simplify
        if opt.simplify:
            try:
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                        dynamic_input_shape=opt.dynamic,
                                                        input_shapes={'images': list(img.shape)} if opt.dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, opt.output_onnx)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {opt.output_onnx} ({file_size(opt.output_onnx):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

