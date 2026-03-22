import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import utils.transforms as trans
from torchvision import transforms
import time
from torch.utils import data
import numpy as np
import os

import data.dataset as dataset_loader
from utils.utils import save_depth_quality, save_edge


def inference(net, args, save_edge_flag=False, save_depth_quality_flag=False):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model weights from: {args.save_model_dir}")
    try:
        state_dict = torch.load(args.save_model_dir, map_location=device)
        net.load_state_dict(state_dict)
        net = net.to(device)
        net.eval()
        print("[INFO] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    datasets_paths = args.test_paths.split('+')
    print(f"[INFO] Test datasets: {datasets_paths}")

    for dataset_name in datasets_paths:
        dataset = dataset_loader.get_loader_pp(
            dataset_name,
            os.path.join(args.data_root, "TestDataset/"),
            args.img_size,
            mode="test"
        )
        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        print(f"\n[INFO] Inference on dataset: {dataset_name}, {len(dataloader)} image(s)")

        time_list = []

        with torch.no_grad():
            for idx, data_batch in enumerate(dataloader):
                images, depths, image_w, image_h, image_path = data_batch
                images = Variable(images.to(device))
                depths = Variable(depths.to(device))

                start_time = time.time()
                predict_res = net(images, depths)
                infer_time = time.time() - start_time
                time_list.append(infer_time)

                mask = predict_res[0][-1]
                image_w, image_h = int(image_w[0]), int(image_h[0])

                transform = trans.Compose([
                    transforms.ToPILImage(),
                    trans.Scale((image_w, image_h))
                ])
                output = F.sigmoid(mask)
                output = output.data.cpu().squeeze(0)
                output = transform(output)

                save_pred_root = os.path.join(args.save_test_path_root, dataset_name )
                os.makedirs(save_pred_root, exist_ok=True)
                filename = image_path[0].split('/')[-1].split('.')[0]
                save_path = os.path.join(save_pred_root, f"{filename}.png")

                output.save(save_path)
                if idx % 1 == 0:
                    print(f"[PROGRESS] {idx + 1}/{len(dataloader)} - saved to: {save_path}")
                if save_edge_flag:
                    save_edge(save_pred_root, filename, predict_res[1][-1], transform)
                if save_depth_quality_flag:
                    save_depth_quality(save_pred_root, filename, predict_res[2][-1], transform)
                break
        avg_time = np.mean(time_list)
        fps = 1 / avg_time if avg_time > 0 else 0
        print(f"\n[INFO] Finished dataset: {dataset_name}")
        print(f"Average time per image: {avg_time:.4f}s, FPS: {fps:.2f}")

    print("\n[INFO] All datasets completed.")


if __name__ == "__main__":
    from utils.options import config
    args = config()
    from Models.DPPNet import DPPNet
    net = DPPNet(args)
    inference(net, args)