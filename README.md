# learning
what I learnt of a project that I joined ：test.py 中的代码和相应的注释和理解
这段代码主要用于图像分割，使用了一个名为BiSeNet的模型。以下是代码的主要部分的注释：

# 导入所需的库和模块
from face_logger import setup_logger
from model import BiSeNet
import torch
import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# 定义一个函数，用于可视化分割结果
def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return vis_im



# 定义一个函数，用于评估模型的分割效果
def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    # 创建结果保存的目录
    if not os.path.exists(respth):
        os.makedirs(respth)

    # 初始化模型
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    save_pth = osp.join('model', cp)
    # 加载模型权重
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    # 设置模型为评估模式
    net.eval()

    # 定义图像预处理操作
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

  # 对指定目录下的每一张图像进行分割，并保存结果
  with torch.no_grad():
      for image_path in os.listdir(dspth):
          img = Image.open(osp.join(dspth, image_path))
          image = img.resize((512, 512), Image.BILINEAR)
          img = to_tensor(image)
          img = torch.unsqueeze(img, 0)
          # img = img.cuda()
          out = net(img)[0]
          parsing = out.squeeze(0).cpu().numpy().argmax(0)
          # print(parsing)
          print(np.unique(parsing))
          
          vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))


# 主函数，调用evaluate函数进行图像分割
if __name__ == "__main__":
    evaluate(dspth='makeup', cp='79999_iter.pth')
