import os
import argparse
import yaml

import matplotlib.pyplot as plt
import torch as th
import torchaudio

from models import *


class VADPredictor(object):
    def __init__(self, ckpt_folder, weight_folder, device) -> None:
        super().__init__()
        self.ckpt_folder = ckpt_folder
        self.cfg = yaml.load(open(os.path.join(ckpt_folder, 'hparams.yml')), Loader=yaml.FullLoader)
        self.weight_folder = weight_folder
        self.device = device
        self.__model = self.load_model(weight_folder)
        self.EPS = 1e-8

    def load_model(self, weight_folder):
        model = VADNet(**self.cfg['model'])
        state_dict = th.load(weight_folder, weights_only=False)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def get_mel_spec(self, input_audio):
        audio, _ = torchaudio.load(input_audio)
        melspec_params = {
            'sample_rate':  self.cfg['data']['sr'],
            'n_fft' : self.cfg['data']['nfft'],
            'hop_length' :  self.cfg['data']['hop_length'],
            'n_mels' : self.cfg['data']['n_mels'],
        }
        melspec = torchaudio.transforms.MelSpectrogram(**melspec_params)
        mel = th.log(melspec(audio)+self.EPS).to(self.device)
        return mel

    def predict(self, audio_path, threshold=None):
        mel = self.get_mel_spec(audio_path)
        with th.no_grad():
            probs = th.sigmoid(self.__model.forward(mel.unsqueeze(0)))
            probs = probs[0, :, 0].cpu().numpy()
        
        if threshold is not None:
            probs[probs >= threshold] = 1
            probs[probs < threshold] = 0

        return probs

    def get_model(self):
        return self.__model
    
    def plot_result(self, audio_path, threshold=None, save = False, save_dir="pics", filename="result.png") -> None:
        mel = self.get_mel_spec(audio_path)
        probs = self.predict(audio_path, threshold)

        plt.figure(figsize=(10, 6))
        plt.plot(probs * 40, color='red', label="Prediction")
        mel_cpu = mel.cpu().numpy()
        plt.pcolormesh(mel_cpu[0], cmap='viridis', alpha=0.6)
        plt.colorbar(label="Mel Spectrogram (log-scaled)")
        # plt.title("Voice Detection Result")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.legend()
        
        if save:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename).replace('\\', '/')
            plt.savefig(save_path)
            print(f"Результат сохранён в файл: {save_path}")
        
        plt.show() 
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', default='data/samples/5ed8a1c0f3ea2.flac', type=str, help='path to file to predict VAD')
    parser.add_argument('-plot', '--plot_result', action='store_true', default=False, help='Plot spectrogram and model predictions')
    parser.add_argument('-s', '--save_result', action='store_true', default=False, help='Save spectogram and model predections')
    parser.add_argument('-t', '--threshold', type=float, default=None, help='threshold value')
    parser.add_argument('-c', '--ckpt_folder', default='./checkpoints/128_mels', help='Path to model checkpoint')
    parser.add_argument('-w', '--weight_folder', default='./weights/VADNet_30_0.0281_0.9889.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    
    predictor = VADPredictor(ckpt_folder=args.ckpt_folder, weight_folder = args.weight_folder, device='cuda')

    if not args.plot_result:
        probs = predictor.predict(audio_path=args.input_file, threshold=args.threshold)
    else: 
        probs = predictor.plot_result(audio_path=args.input_file, threshold=args.threshold, save = args.save_result)
