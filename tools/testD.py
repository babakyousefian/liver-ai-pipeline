import torch

ckpt = torch.load(
    r"C:\Users\liana\Documents\pythonProjects\FinalTasks\finalReports\bestModels\bestKDEnsemble\bestKDE.pth",
    map_location="cpu",
    weights_only=False,
)
print("Top keys:", type(ckpt), list(ckpt.keys()) if isinstance(ckpt, dict) else "not dict")
sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt.state_dict()
for k, v in list(sd.items())[-15:]:
    print(f"  {k}: {tuple(v.shape)}")