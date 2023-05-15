import torch as th
from advfussion.data_model.my_loader import MyCustomDataset
from robustbench.utils import load_model
from tqdm import tqdm

# hyper
batch_size = 10
name = "Salman2020Do_50_2"
threat_model = "Linf"

# main

data = MyCustomDataset(img_path="../data/images")
sampler = th.utils.data.SequentialSampler(data)
attack_loader = th.utils.data.DataLoader(dataset=data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            sampler=sampler, num_workers=2, pin_memory=True)
attack_model = load_model(model_name=name, dataset='imagenet', threat_model=threat_model)
attack_model = attack_model.cuda()
natural_err_total = th.tensor(0.0).cuda()
count = 0
p_bar = tqdm(attack_loader)
for img, label, img_name in p_bar:
    img, label = img.cuda(), label.cuda()
    # get natural_err_total
    count += len(img)
    with th.no_grad():
        err = (attack_model(img).data.max(1)[1] != label.data).float().sum()
        natural_err_total += err
    p_bar.set_description("natural_err_total: {}".format(natural_err_total/count))

print(natural_err_total/count)