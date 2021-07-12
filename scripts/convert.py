import torch
from collections import OrderedDict



if __name__ == "__main__":
    model = torch.load("checkpoint21_110000.pth")
    m = model["model"]
    m_t = OrderedDict()
    for k in m.keys():
        m_t[k[7:]] = m[k]
    model["model"] = m_t
    torch.save(model, "checkpoint.pth")

        
    
