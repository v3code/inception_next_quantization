import torch
from tqdm import tqdm


def calibrate_model(model, dataloader):
    for batch in tqdm(dataloader, desc='Calibrating...'):
        model(batch)


def configure_model(model, qconfig):
    model.eval()
    model.qconfig = qconfig


def test_calibrate_model(model, image_size=224, num_samples=4):
    model(torch.randn(num_samples, 3, image_size, image_size))


def static_quantize_model_fx(model, samples):
    raise NotImplementedError


def static_quantize_model(model_fp32, calibrator):
    model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
    calibrator(model_fp32_prepared)
    return torch.ao.quantization.convert(model_fp32_prepared)


def dynamically_quantize_model(model_fp32):
    raise NotImplementedError
