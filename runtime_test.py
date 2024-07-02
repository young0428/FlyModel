import torch
from VCL import VCL, VCL_Trainer
import time



if __name__ == '__main__':
    torch.cuda.empty_cache()
    batch_size = 1
    frame_num = 5
    h = 64
    w = 128
    c = 5

    lr = 1e-6
    input_dims = 5 # up, down, right, left, origin
    model = VCL(input_dims = input_dims, video_size=(h, w))
    trainer = VCL_Trainer(model, lr)


    total_training_time = 0
    for iter in range(40):
        test_input = torch.randn(batch_size, frame_num, h, w, c).half() # B, T, H, W, C
        test_output = torch.randn(batch_size, frame_num).half()

        start_time = time.time()
        trainer.step(test_input, test_output)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Traning {iter + 1}, elapsed time: {elapsed_time:.3f}')
        total_training_time += elapsed_time

      # 경과 시간 계산
    print(f"Total execution time: {total_training_time:.3f} seconds")
    exit()

