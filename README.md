# DDIS

Official repository for the ICML 2025 paper:

**When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class**

- Paper: https://arxiv.org/abs/2506.15381
- Code: https://github.com/kdst-team/DDIS

## Overview

DDIS is a diffusion-assisted data-free image synthesis method that uses a
text-to-image diffusion model as a strong image prior for Data-Free Image
Synthesis (DFIS). The method extracts knowledge from a pretrained classifier
and guides the diffusion process so that synthesized samples align better with
the original training distribution.

The framework introduces:

- **Domain Alignment Guidance (DAG)** to align generated images with the target
  domain during diffusion sampling.
- **Class Alignment Token (CAT)** to capture class-specific semantics from the
  given pretrained model.

Experiments in the paper are reported on PACS and ImageNet.

## Repository Layout

- `index.html`, `static/`, and the figure assets in the repository root:
  project website for the paper.
- `DDIS_code/main_ddis.py`: main training, evaluation, and control-strength
  entry point.
- `DDIS_code/config.py`: experiment configuration.
- `DDIS_code/utils.py`: classifier and diffusion-model setup utilities.
- `DDIS_code/p2p/`: expected location of Prompt-to-Prompt related helper files.

## Setup Notes

This repository currently does not ship a dedicated environment file. The code
uses Python with the following libraries, as referenced in the source:

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `kornia`
- `numpy`
- `matplotlib`
- `pyrallis`

Depending on the experiment setup, additional pretrained checkpoints may also
be required.

## Running the Code

The main entry point is:

```bash
cd DDIS_code
python3 main_ddis.py --class_index 1 --train True --evaluate False --control_strength False
```

There is also a simple loop script:

```bash
cd DDIS_code
bash run.sh <cuda_device_id>
```

For experiment-specific settings, edit `DDIS_code/config.py` and inspect
`DDIS_code/main_ddis.py`.

## Important Note on `p2p`

To keep fresh clones and GitHub Actions checkouts stable, the previously broken
submodule entry at `DDIS_code/p2p` was removed.

If you want to run `DDIS_code/main_ddis.py`, you should place the expected
Prompt-to-Prompt source files in `DDIS_code/p2p/`, including:

- `ptp_utils_distG.py`
- `prompt_to_prompt.py`

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2025ddis,
  title={When Model Knowledge meets Diffusion Model: Diffusion-assisted Data-free Image Synthesis with Alignment of Domain and Class},
  author={Kim, Yujin and Kim, Hyunsoo and Kim, Hyunwoo J. and Kim, Suhyun},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## License

This repository includes an MIT license in [LICENSE](LICENSE).
