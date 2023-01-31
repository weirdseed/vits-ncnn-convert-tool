import argparse
import os
import re
import torch
from utils import get_hparams_from_file, load_checkpoint
from models_ncnn import SynthesizerTrn
import shutil
from torch import _weight_norm
from torch.nn import Parameter

def create_folders(model_path, multi):
    # creating dirs
    cache_root = "ncnn_cache/" # temp folder
    model_path = model_path.replace("\\","/")
    match = re.match('.*/(.*)\.pth', model_path)
    out_folder = match.group(1) # out folder

    # flow
    flow_folder = cache_root + "flow"

    # flow reversed
    flow_reversed_folder = cache_root + "flow_reverse"

    # enc_p folder
    enc_p_folder = cache_root + "enc_p"

    # dp
    dp_folder = cache_root + "dp"

    # dec
    dec_folder = cache_root + "dec"

    # enc_q
    enc_q_folder = cache_root + "enc_q"

    folders = [cache_root, out_folder, flow_reversed_folder, flow_folder,  enc_p_folder, dp_folder, dec_folder, enc_q_folder]

    if not multi:
        folders.remove(flow_folder)
        folders.remove(enc_q_folder)
    
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)
            if not os.path.exists(folder): 
                raise RuntimeError("Directory creation failed!")
    return folders

def convert_model(net, folder, name, multi):
    layer_inputs = None
    if multi:
        layer_inputs = {
            "enc_p": [torch.randint(0,20,(1,100)), torch.LongTensor([100]), net.enc_p.emb.weight.data],
            "dp": [torch.randn((1,192,100)),torch.ones((1,1,100)),torch.randn((1, 2, 100)),0.8 * torch.ones((1,2,100)),torch.randn((1,256,1))],
            "flow":[torch.randn((1,192,255)),torch.ones((1,1,255)),torch.randn((1,256,1))],
            "flow.reverse": [torch.randn((1,192,255)),torch.ones((1,1,255)),torch.randn((1,256,1))],
            "dec": [torch.randn((1,192,255)),torch.randn((1,256,1))],
            "enc_q": [torch.randn((1,513,336)),torch.LongTensor([336]),torch.randn((1,256,1))]
        }
    else:
        layer_inputs = {
            "enc_p": [torch.randint(0,20,(1,100)), torch.LongTensor([100]), net.enc_p.emb.weight.data],
            "dp": [torch.randn((1,192,100)),torch.ones((1,1,100)),torch.randn((1, 2, 100)),0.8 * torch.ones((1,2,100))],
            "flow.reverse": [torch.randn((1,192,255)),torch.ones((1,1,255))],
            "dec": [torch.randn((1,192,255))],
        }
    custom_ops = {
        "enc_p": "modules.Transpose,modules.SequenceMask,modules.Embedding,attentions.Attention,attentions.ExpandDim,attentions.SamePadding",
        "dp": "modules.PRQTransform,modules.Transpose,modules.ReduceDims",
        "flow": "modules.ResidualReverse",
        "flow.reverse": "modules.ResidualReverse",
        "dec": "",
        "enc_q": "modules.RandnLike,modules.ResidualReverse,modules.SequenceMask"
    }
    
    if name == "flow_reverse":
        name = name.replace("_",".")
        layer = getattr(net, "flow")
        layer.reverse = True
    elif name == "flow":
        layer = getattr(net, name)
        layer.reverse = False
    else:
        layer = getattr(net, name)
    path_pt = os.path.join(folder, name+".pt")
    torch.jit.trace(layer, layer_inputs[name]).save(path_pt)
    os.system("{} {} fp16={} moduleop={}".format(pnnx_path, path_pt, 1 if fp16 else 0, custom_ops[name]))

def export(root_folder, out_folder, name):
    # copy files
    src_folder = os.path.join(root_folder, name)
    if name == "flow_reverse":
        name = name.replace("_",".")

    src_path = os.path.join(src_folder, name+".ncnn.bin")
    target_path = os.path.join(out_folder, name+".ncnn.bin")
    shutil.copy(src_path, target_path)

def main(args):
    # multi model or not
    multi = False
    global pnnx_path
    global fp16

    # input path
    config_path = args.config_path
    model_path = args.model_path
    fp16 = args.fp16

    if os.name == "nt":
        pnnx_path = "pnnx\\pnnx.exe"
    elif os.name == "posix":
        pnnx_path = "pnnx/pnnx"
    else:
        raise RuntimeError("unsupported system!")

    if not os.path.exists(config_path):
        raise RuntimeError("Config file does not exist!")

    if not os.path.exists(model_path):
        raise RuntimeError("Model file does not exist! Currently only \"japanese_leaners\" and \"japanese_leaners2\" are supported.")

    if not os.path.exists(pnnx_path):
        raise RuntimeError("pnnx does not exist!")


    # load configs
    hps = get_hparams_from_file(config_path)
    n_symbols = len(hps.symbols) if 'symbols' in hps.keys() else 0

    if n_symbols == 0:
        raise RuntimeError("Symbols can not be empty!")
        
    for cleaner in hps.data.text_cleaners:
        if cleaner not in ["japanese_cleaners","japanese_cleaners2"]:
            raise RuntimeError("This cleaner is not supported!")

    if hps.data.n_speakers > 0:
        multi = True

    # create model
    if multi:
        net_g = SynthesizerTrn(
        n_symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    else:
        net_g = SynthesizerTrn(
            n_symbols,
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)

    # load checkpoints
    _ = net_g.eval()
    _ = load_checkpoint(model_path,net_g, None)

    # remove redundant weigths
    for k, module in net_g.named_modules():
        try:
            g = getattr(module, "weight_g")
            v = getattr(module, "weight_v")
            normed = _weight_norm(v, g, 0)
            module.weight = Parameter(normed)
            delattr(module, "weight_g")
            delattr(module, "weight_v")
        except Exception:
            continue
    
    # create folders
    folders = create_folders(model_path, multi)
    cache_root = folders[0]
    out_root = folders[1]

    # convert
    for folder in folders[2:]:
        name = folder.replace(cache_root, "")
        convert_model(net_g, folder, name, multi)

    # export embedding
    emb_weight = net_g.enc_p.emb.weight.data.flatten().numpy().astype("float32")
    with open(os.path.join(out_root, "emb_t.bin"), "wb") as f:
        f.write(emb_weight)

    emb_weight = net_g.emb_g.weight.data.flatten().numpy().astype("float32")
    with open(os.path.join(out_root, "emb_g.bin"), "wb") as f:
        f.write(emb_weight)

    # export
    for folder in folders[2:]:
        name = folder.replace(cache_root, "")
        export(cache_root, out_root, name)

    # clean
    shutil.rmtree(cache_root)
    if os.path.exists("debug.bin"):
        os.remove("debug.bin")

    if os.path.exists("debug.param"):
        os.remove("debug.param")

    if os.path.exists("debug2.bin"):
        os.remove("debug2.bin")
    
    if os.path.exists("debug2.param"):
        os.remove("debug2.param")
    print("Cleaned!")

    shutil.copy(config_path, os.path.join(out_root,"config.json"))
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config_path",type=str, help="path/to/config.json")
    parser.add_argument("-m", "--model_path", type=str, help="path/to/model.pth")
    parser.add_argument("-fp16", "--fp16", action="store_true", help="half precision on/off")
    args = parser.parse_args()
    main(args)