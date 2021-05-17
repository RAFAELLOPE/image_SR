import matplotlib.pylab as plt
import numpy as np
from model import DCSRN
import torch
from get_data import Data


class SREvaluator:
    def __init__(self, hr_img, lr_img, model_path="/model/model.pt", device="cpu"):
        self.device = device
        self.hr_img = hr_img.unsqueeze(0).to(device)
        self.lr_img = lr_img.unsqueeze(0).to(device)
        self.checkpoint = torch.load(model_path)
        
        n_channels = self.checkpoint['N_channels']
        model = DCSRN(n_channels)
        model.load_state_dict(self.checkpoint['model_state_dict']).to(device)
        model.eval()
        
        self.training_err_list = self.checkpoint['Training loss']
        self.validation_err_list = self.checkpoint['Validation loss']
        self.hr_hat_img = self.model(self.lr_img).to(device)
        
    
    def plot_outputs(self, slice=0):

        hr = self.hr_img.detach().numpy()[0]
        lr = self.lr_img.detach().numpy()[0]
        hr_out = self.hr_hat_img[0].detach().numpy()[0]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
        axs = axs.flatten()
        axs[0].imshow(self.hr[:,:,slice], cmap='gray')
        axs[0].set_title('HR Image')
        axs[1].imshow(self.lr[:,:,slice], cmap='gray')
        axs[1].set_title('LR Image')
        axs[2].imshow(self.hr_out[:,:,slice], cmap='gray')
        axs[2].set_title('HR output Image')
        plt.savefig('image_comparison.png')

    def plot_train_val_output(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.training_err_list, color='red', label='training')
        plt.plot(self.validation_err_list, color='blue', label='validation')
        plt.legend()
        plt.savefig('train_test_error.png')

    def compute_psnr(self):
        hr = self.hr_img.detach().numpy()[0]
        hr_out = self.hr_hat_img[0].detach().numpy()[0]
        mse = np.sqrt(np.sum((hr - hr_out)**2))
        psnr = (np.max(self.hr))/mse
        psnr_db = 20*np.log10(psnr)
        with open('report.txt', 'rw') as report:
            report.save(f"PSNR: {psnr}\n PSNR (db):{psnr_db}")

if __name__ == "__main__":
    test_dataset = Data(train=False)
    lr_img, hr_img = test_dataset[0]
    evaluator = SREvaluator(hr_img= hr_img,
                            lr_img = lr_img)
    
    evaluator.plot_outputs(slice=5)
    evaluator.compute_psnr()
    evaluator.plot_train_val_output()

