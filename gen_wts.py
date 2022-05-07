import torch
import struct
 
net = torch.load('yolox_s.pth',map_location='cpu')	# 加载pth文件，pth_path为pth文件的路径、
# print(net)
# for k in net.keys():			# 打印net的key
#   print(k)
  
# print(net['amp'])		# 我的为start_epoch,model,optimizer,amp
# 显然，模型参数保存在 net["model_state_dict"]，有时候命名会不一样，所以应打印net的key确定一下
model_state_dict = net["model"]
f = open("yolox.wts", 'w')  # 自己命名wts文件
f.write("{}\n".format(len(model_state_dict.keys())))  # 保存所有keys的数量

for k, v in model_state_dict.items():
    vr = v.reshape(-1).cpu().numpy()	# 权重参数展开成1维
    f.write("{} {}".format(k, len(vr)))  # 保存每一层名称和参数长度
    for vv in vr:
        f.write(" ")
        f.write(struct.pack(">f", float(vv)).hex())  # 使用struct把权重封装成字符串
    f.write("\n")

