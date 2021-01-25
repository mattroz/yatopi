import torch
import onnx
from model import LPRNet, LPRNetv2, LPRNetv3, LPRNet_CRNN_v2
from argparse import ArgumentParser

argp = ArgumentParser()
argp.add_argument('--weights', type=str, default=None, help='Path to model weights')
argp.add_argument('--prefix', type=str, required=True)
args = argp.parse_args()

#m = LPRNetv3(8)
m = LPRNet_CRNN_v2(hidden_size=256,
                           vocab_size=37,
                           bidirectional=True,
                           dropout=0.1)
#inp = torch.randn(1,3,64,128)
inp = torch.randn(1, 3, 50, 200)

if args.weights:
    cpkt = torch.load(args.weights,  map_location=torch.device('cpu') )
    m.load_state_dict(cpkt['model_state_dict'])

torch.onnx.export(m, inp, f'{args.prefix}.onnx')
print('\nCreating onnx model...\n')

print('\nLoading onnx graph...\n')
m_onnx = onnx.load(f'{args.prefix}.onnx')
print(onnx.helper.printable_graph(m_onnx.graph))

n_params = sum(p.numel() for p in m.parameters())
print('Number of parameters: ', n_params)



