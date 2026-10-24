import torch

def save_checkpoint(model, optimizer, iteration, out) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(
        src,
        map_location="cpu",  # 控制 tensor 加载到哪个设备。如果 torch.load尝试放到GPU,在没有GPU的机器就会报错，强制把所有 tensor 都放到 CPU 上，与保存时的设备无关。这样在任意机器上都能加载，之后再用 model.to(device) 搬到目标设备
        weights_only=True,# 只允许反序列化白名单类型（tensor、基本 Python 类型、字典、列表等"纯数据"）
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

