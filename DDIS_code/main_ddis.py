import torch

import os
from pathlib import Path
import torch.utils.checkpoint
import torchvision.utils as vutils
import itertools
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
#import DF_synthesis_LDM.StableDiffusionImg2ImgPipeline_running_stat_guide_ver2 as StableDiffusionImg2ImgPipeline_running_stat_guide_ver2
import prompt_dataset
import utils
from inet_classes import IDX2NAME as IDX2NAME_INET
import torchvision
import matplotlib.pyplot as plt
import numpy as np
#pipe도 바꾸고 ptp_utils_distG로 바꿔야 하지 않을까? 모두 다 바꾸고 실험을 한번 해보자.
#import p2p.ptp_utils as ptp_utils
import p2p.ptp_utils_distG as ptp_utils
import p2p.prompt_to_prompt as protpro


from config import RunConfig
import pyrallis
import shutil
import pacs_classes
import cifar10_classes
import cifar100_classes

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = '1'
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()

def train(config: RunConfig):
    # A range of imagenet classes to run on
    start_class_idx = config.class_index
    stop_class_idx = config.class_index

    # Classification model
    classification_model = utils.prepare_classifier(config)

    current_early_stopping = RunConfig.early_stopping

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    if "inet" in config.classifier:
        IDX2NAME = IDX2NAME_INET
    elif 'pacs' in config.classifier:
        IDX2NAME = pacs_classes.IDX2NAME
    elif 'cifar10_resnet34' == config.classifier:
        IDX2NAME = cifar10_classes.IDX2NAME
    elif 'cifar100_resnet34' == config.classifier:
        IDX2NAME = cifar100_classes.IDX2NAME
    else:
        IDX2NAME = classification_model.config.id2label

    #### Train ####
    print(f"Start experiment {exp_identifier}")

    for running_class_index, class_name in IDX2NAME.items():
        running_class_index += 1
        if running_class_index < start_class_idx:
            continue
        if running_class_index > stop_class_idx:
            break

        class_name = class_name.split(",")[0]
        print(f"Start training class token for {class_name}")
        img_dir_path = f"img/{config.prefix}*{config.grad_scale}r_grad{config.lambda_bn}_{class_name}/train"
        if Path(img_dir_path).exists():
            shutil.rmtree(img_dir_path)
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        # Stable model
        unet, vae, text_encoder, scheduler, tokenizer,_ = utils.prepare_stable(config)

        #  Extend tokenizer and add a discriminative token ###
        class_infer = config.class_index - 1
        prompt_suffix = " ".join(class_name.lower().split("_"))

        ## Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(config.placeholder_token)
        print("tokenzier length : ",len(tokenizer))
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {config.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        ## Get token ids for our placeholder and initializer token.
        # This code block will complain if initializer string is not a single token
        ## Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(config.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(config.placeholder_token)

        # we resize the token embeddings here to account for placeholder_token
        text_encoder.resize_token_embeddings(len(tokenizer))

        #  Initialise the newly added placeholder token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Define dataloades

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            input_ids = tokenizer.pad(
                {"input_ids": input_ids}, padding=True, return_tensors="pt"
            ).input_ids
            texts = [example["instance_prompt"] for example in examples]
            batch = {
                "texts": texts,
                "input_ids": input_ids,
            }
            return batch

        train_dataset = prompt_dataset.PromptDataset(
            prompt_suffix=prompt_suffix,
            tokenizer=tokenizer,
            placeholder_token=config.placeholder_token,
            number_of_prompts=config.number_of_prompts,
            epoch_size=config.epoch_size,
        )

        train_batch_size = config.batch_size
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Define optimization

        ## Freeze vae and unet
        utils.freeze_params(vae.parameters())
        utils.freeze_params(unet.parameters())

        ## Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        utils.freeze_params(params_to_freeze)

        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=config.lr,
            betas=config.betas,
            weight_decay=config.weight_decay,
            eps=config.eps,
        )
        criterion = torch.nn.CrossEntropyLoss()

        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

        if config.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        text_encoder, optimizer, train_dataloader = accelerator.prepare(
            text_encoder, optimizer, train_dataloader
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)

        classification_model = classification_model.to(accelerator.device)
        text_encoder = text_encoder.to(accelerator.device)

        # Keep vae in eval mode as we don't train it
        vae.eval()
        # Keep unet in train mode to enable gradient checkpointing
        unet.train()

        global_step = 0
        total_loss = 0
        min_loss = 99999

        # Define token output dir
        token_dir_path = f"token/{class_name}"
        Path(token_dir_path).mkdir(parents=True, exist_ok=True)
        token_path = f"{token_dir_path}/{exp_identifier}_{class_name}"

        latents_shape = (
            config.batch_size,
            unet.config.in_channels,
            config.height // 8,
            config.width // 8,
        )
        loss_r_feature_layers = []
        classification_model.eval()
        for module in classification_model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        if config.skip_exists and os.path.isfile(token_path):
            print(f"Token already exist at {token_path}")
            return
        else:
            for epoch in range(config.num_train_epochs):
                print(f"Epoch {epoch}")
                generator = torch.Generator(
                    device=config.device
                )  # Seed generator to create the inital latent noise
                generator.manual_seed(config.seed)
                correct = 0
                for step, batch in enumerate(train_dataloader):
                    # setting the generator here means we update the same images
                    classification_loss = None 
                    with accelerator.accumulate(text_encoder):
                        # Get the text embedding for conditioning
                        #encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        encoder_hidden_state = text_encoder(batch["input_ids"])[0]
                        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
                        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
                        # corresponds to doing no classifier free guidance.
                        do_classifier_free_guidance = config.guidance_scale > 1.0

                        # get unconditional embeddings for classifier free guidance
                        if do_classifier_free_guidance:
                            max_length = batch["input_ids"].shape[-1]
                            uncond_input = tokenizer(
                                [""] * config.batch_size,
                                padding="max_length",
                                max_length=max_length,
                                return_tensors="pt",
                            )
                            uncond_embeddings = text_encoder(
                                uncond_input.input_ids.to(config.device)
                            )[0]

                            # For classifier free guidance, we need to do two forward passes.
                            # Here we concatenate the unconditional and text embeddings into
                            # a single batch to avoid doing two forward passes.
                            encoder_hidden_states = torch.cat(
                                [uncond_embeddings, encoder_hidden_state]
                            )
                            #encoder_hidden_states = torch.cat([encoder_hidden_states,encoder_hidden_states])
                        encoder_hidden_states = encoder_hidden_states.to(
                            dtype=weight_dtype
                        )
                        init_latent = torch.randn(
                            latents_shape, generator=generator, device="cuda"
                        ).to(dtype=weight_dtype)

                        latents = init_latent
                        scheduler.set_timesteps(config.num_of_SD_inference_steps)
                        grad_update_step = config.num_of_SD_inference_steps - 1
                        # generate image
                        for i, t in enumerate(scheduler.timesteps):
                            guided_latents = latents
                            #print("latent shape :",latents.shape)
                            if i < grad_update_step:  # update only partial
                                with torch.enable_grad():
                                   #noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_state).sample
                                    #scheduler_output = scheduler.step(noise_pred, t, latents)
                                    # Access the prev_sample attribute
                                    #z_0 = scheduler_output.prev_sample
                                    #print("z_0 shape : ",z_0.shape)
                                    #z_0 = z_0.detach().requires_grad_(True) 
                                   guided_latents = guided_latents.detach().requires_grad_(True)
                                   input_latents = 1/0.18215*guided_latents
                                   image = vae.decode(input_latents).sample
                                   image = (image / 2 + 0.5).clamp(0, 1)
                                
                                   if "cifar10" in config.classifier:
                                    x_in_temp  = torch.nn.functional.interpolate(image, size=32)
                                   else:
                                    x_in_temp  = torch.nn.functional.interpolate(image, size=224)
                                
                                   #print("cifar10 shape :",x_in_temp.shape) # 1 3 32 32
                                   out = classification_model(x_in_temp)
                                   loss_r_feature = sum([mod.r_feature for (idx, mod) in enumerate(loss_r_feature_layers)])#if idx >= 30])
                                   r_grad = torch.autograd.grad(config.lambda_bn*loss_r_feature.sum(), guided_latents)[0]
                            
                                   vutils.save_image(image.detach(),f'./{i}_intermediate image.png',normalize=True, scale_each=True, nrow=int(1))
                                   guided_latents = guided_latents - 0.01*r_grad
                                   # = z_0 - 0.01*r_grad
                                   #latent = latent -0.01*r_grad

                                with torch.no_grad():
                                    latents = latents + config.grad_scale*(guided_latents-latents)
                                    #latents = latents - config.grad_scale*r_grad
                                    latent_model_input = (
                                        #torch.cat([latents,guided_latents])
                                        torch.cat([latents]*2)
                                        if do_classifier_free_guidance
                                        else latents
                                    )
                                    noise_pred = unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=encoder_hidden_states,
                                    ).sample

                                    # perform guidance
                                    if do_classifier_free_guidance:
                                        (
                                            noise_pred_uncond,
                                            noise_pred_text,
                                        ) = noise_pred.chunk(2)
                                        noise_pred = (
                                            noise_pred_uncond
                                            + config.guidance_scale
                                            * (noise_pred_text - noise_pred_uncond)
                                        )
                                    latents = scheduler.step(
                                        noise_pred, t, latents
                                    ).prev_sample
                            else:
                                latent_model_input = (
                                   #torch.cat([latents,guided_latents])
                                    torch.cat([latents]*2)
                                    if do_classifier_free_guidance
                                    else latents
                                )
                                noise_pred = unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=encoder_hidden_states,
                                ).sample
                                # perform guidance
                                if do_classifier_free_guidance:
                                    (
                                        noise_pred_uncond,
                                        noise_pred_text,
                                    ) = noise_pred.chunk(2)
                                    noise_pred = (
                                        noise_pred_uncond
                                        + config.guidance_scale
                                        * (noise_pred_text - noise_pred_uncond) #y에 관한 거니까... guidance를 이렇게 해주는게 맞을것같다
                                    )
                                latents = scheduler.step(
                                    noise_pred, t, latents
                                ).prev_sample
                                # scale and decode the image latents with vae

                        latents_decode = 1 / 0.18215 * latents
                        image = vae.decode(latents_decode).sample
                        image = (image / 2 + 0.5).clamp(0, 1)

                        image_out = image

                        image = utils.transform_img_tensor(image, config)
                        if  config.classifier=="inet_resnet34" or 'pacs' or 'cifar10_resnet34' in config.classifier:
                            output = classification_model(image)
                        else:
                            output = classification_model(image).logits

                        if classification_loss is None:
                            classification_loss = criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )
                        else:
                            classification_loss += criterion(
                                output, torch.LongTensor([class_infer]).cuda()
                            )

                        pred_class = torch.argmax(output).item()
                        total_loss += classification_loss.detach().item()

                        # log
                        txt = f"On epoch {epoch} \n"
                        with torch.no_grad():
                            txt += f"{batch['texts']} \n"
                            txt += f"Desired class: {IDX2NAME[class_infer]}, \n"
                            txt += f"Image class: {IDX2NAME[pred_class]}, \n"
                            txt += f"Loss: {classification_loss.detach().item()}"
                            with open("run_log.txt", "a") as f:
                                print(txt, file=f)
                            print(txt)
                            utils.numpy_to_pil(
                                image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
                            )[0].save(
                                f"{img_dir_path}/{epoch}_{IDX2NAME[pred_class]}_{classification_loss.detach().item()}.jpg",
                                "JPEG",
                            )

                        if pred_class == class_infer:
                            correct += 1

                        torch.nn.utils.clip_grad_norm_(
                            text_encoder.get_input_embeddings().parameters(),
                            config.max_grad_norm,
                        )
                        accelerator.backward(classification_loss)
                        #classification_loss.backward()

                        # Zero out the gradients for all token embeddings except the newly added
                        # embeddings for the concept, as we only want to optimize the concept embeddings
                        if accelerator.num_processes > 1:
                            grads = (
                                text_encoder.module.get_input_embeddings().weight.grad
                            )
                        else:
                            grads = text_encoder.get_input_embeddings().weight.grad

                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero = (
                            torch.arange(len(tokenizer)) != placeholder_token_id
                        )
                        grads.data[index_grads_to_zero, :] = grads.data[
                            index_grads_to_zero, :
                        ].fill_(0)

                        optimizer.step()
                        optimizer.zero_grad()

                        text_encoder.get_input_embeddings().weight.data = torch.clamp(text_encoder.get_input_embeddings().weight.data,-0.1,0.1)

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            if total_loss > 2 * min_loss:
                                print("training collapse, try different hp")
                                config.seed += 1
                                print("updated seed", config.seed)
                            print("update")
                            if total_loss < min_loss:
                                min_loss = total_loss
                                current_early_stopping = config.early_stopping
                                # Create the pipeline using the trained modules and save it.
                                accelerator.wait_for_everyone()
                                if accelerator.is_main_process:
                                    print(
                                        f"Saved the new discriminative class token pipeline of {class_name} to pipeline_{token_path}"
                                    )
                                    if config.sd_2_1:
                                        pretrained_model_name_or_path = (
                                            "stabilityai/stable-diffusion-2-1-base"
                                        )
                                    else:
                                        pretrained_model_name_or_path = (
                                            "CompVis/stable-diffusion-v1-4"
                                        )
                                    pipeline = StableDiffusionPipeline.from_pretrained(
                                        pretrained_model_name_or_path,
                                        text_encoder=accelerator.unwrap_model(
                                            text_encoder
                                        ),
                                        vae=vae,
                                        unet=unet,
                                        tokenizer=tokenizer,
                                    )
                                    pipeline.save_pretrained(f"pipeline_{token_path}")
                            else:
                                current_early_stopping -= 1
                            print(
                                f"{current_early_stopping} steps to stop, current best {min_loss}"
                            )

                            total_loss = 0
                            global_step += 1
                print(f"Current accuracy {correct / config.epoch_size}")

                if (correct / config.epoch_size >= 0.7) or current_early_stopping < 0:
                    break


def evaluate_domain(config: RunConfig):
    class_index = config.class_index - 1
    classification_model = utils.prepare_classifier(config)

    if "inet" in config.classifier:
        IDX2NAME = IDX2NAME_INET
    elif 'pacs' in config.classifier:
        IDX2NAME = pacs_classes.IDX2NAME
    elif 'cifar10_resnet34' == config.classifier:
        IDX2NAME = cifar10_classes.IDX2NAME
    elif 'cifar100_resnet34' == config.classifier:
        IDX2NAME = cifar100_classes.IDX2NAME
    else:
        IDX2NAME = classification_model.config.id2label

    class_name = IDX2NAME[class_index].split(",")[0]

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    tokens_to_try = [config.placeholder_token]
    # Create eval dir
    img_dir_path = f"img/{config.prefix}*{config.grad_scale}r_grad{config.lambda_bn}_{class_name}/eval"
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)
            tokens_to_try.append(config.initializer_token)
    else:
        tokens_to_try.append(config.initializer_token)

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    prompt_suffix = " ".join(class_name.lower().split("_"))

    for descriptive_token in tokens_to_try:
        correct = 0
        prompt = f"A photo of {descriptive_token} {prompt_suffix}"
        print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_size):
            if descriptive_token == config.initializer_token:
                img_id = f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                if os.path.exists(f"{img_id}_correct.jpg") or os.path.exists(
                    f"{img_id}_wrong.jpg"
                ):
                    print(f"Image exists {img_id} - skip generation")
                    if os.path.exists(f"{img_id}_correct.jpg"):
                        correct += 1
                    continue
            generator = torch.Generator(
                device=config.device
            )  # Seed generator to create the inital latent noise
            generator.manual_seed(seed)
            image_out = pipe(prompt, output_type="pt", generator=generator)[0]
            image = utils.transform_img_tensor(image_out, config)

            if  config.classifier=="inet_resnet34" or 'pacs' or "cifar10" in config.classifier:
                output = classification_model(image)
            else:
                output = classification_model(image).logits

            pred_class = torch.argmax(output).item()

            if descriptive_token == config.initializer_token:
                img_path = (
                    f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
            else:
                img_path = (
                    f"{img_dir_path}/{seed}_{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                )

            utils.numpy_to_pil(image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[
                0
            ].save(img_path, "JPEG")

            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )

def evaluate(config: RunConfig):
    to_tensor = torchvision.transforms.ToTensor()

    class_index = config.class_index - 1

    classification_model = utils.prepare_classifier(config)
    classification_model = classification_model.cuda()
    classification_model.eval()

    if "inet" in config.classifier:
        IDX2NAME = IDX2NAME_INET
    elif 'pacs' in config.classifier:
        IDX2NAME = pacs_classes.IDX2NAME
    elif 'cifar10_resnet34' == config.classifier:
        IDX2NAME = cifar10_classes.IDX2NAME
    elif 'cifar100_resnet34' == config.classifier:
        IDX2NAME = cifar100_classes.IDX2NAME
    else:
        IDX2NAME = classification_model.config.id2label

    class_name = IDX2NAME[class_index].split(",")[0]

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )

    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    print("pipe path: ", pipe_path)
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)
    
    #pipe = StableDiffusionImg2ImgPipeline_running_stat_guide_ver2.StableDiffusionImg2ImgPipeline.from_pretrained(pipe_path)
    #pipe = pipe.to(config.device)
    #images = pipe(prompt=prompt, guidance_scale=7.5, classifier = classification_model).images

    tokens_to_try = [config.placeholder_token]
    # Create eval dir
    #img_dir_path = f"img/{config.prefix}/{class_name}/eval"
    img_dir_path = f"img/{config.prefix}*{config.grad_scale}r_grad{config.lambda_bn}_{class_name}/eval"
    dataset_path = f"./syn_{config.prefix}/{class_name}"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    '''
    if Path(img_dir_path).exists():
        print("Img path exists {img_dir_path}")
        if config.skip_exists:
            print("baseline exists - skip it. Set 'skip_exists' to False regenerate.")
        else:
            shutil.rmtree(img_dir_path)
            tokens_to_try.append(config.initializer_token)
    else:
        tokens_to_try.append(config.initializer_token)'
    '''

    Path(img_dir_path).mkdir(parents=True, exist_ok=True)
    prompt_suffix = " ".join(class_name.lower().split("_"))
    
    for descriptive_token in tokens_to_try:
        correct = 0
        prompt = f"A {descriptive_token} {prompt_suffix}"
        print(f"Evaluation for the prompt: {prompt}")

        for seed in range(config.test_start_size,config.test_size):
            print(str(seed)+"-th image sampling")
            if descriptive_token == config.initializer_token:
                img_id = f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                if os.path.exists(f"{img_id}_correct.jpg") or os.path.exists(
                    f"{img_id}_wrong.jpg"
                ):
                    print(f"Image exists {img_id} - skip generation")
                    if os.path.exists(f"{img_id}_correct.jpg"):
                        correct += 1
                    continue
            generator = torch.Generator(
                device=config.device
            )  # Seed generator to create the inital latent noise
            generator.manual_seed(seed)
            controller = protpro.AttentionStore()
            image_pt, image_np, x_t = ptp_utils.text2image_ldm_stable(pipe, [prompt], controller, latent=None, num_inference_steps=config.num_of_SD_inference_steps, guidance_scale= config.guidance_scale, generator=generator, low_resource=False, classification_model= classification_model)
            #ptp_utils.view_images(image_np, num_rows=1, offset_ratio=0.02, prefix=img_dir_path,postfix=f"/{seed}_{descriptive_token}_generated_img")
            
            if "cifar" in config.classifier:
                images = torch.nn.functional.interpolate(image_pt, size=32).to(config.device)
            else:
                images = torch.nn.functional.interpolate(image_pt, size=256).to(config.device)

            if  config.classifier=="inet_resnet34" or 'pacs' or "cifar" in config.classifier:
                output = classification_model(images)
            else:
                output = classification_model(images).logits
            pred_class = torch.argmax(output).item()

            if descriptive_token == config.initializer_token:
                img_path = (
                    f"{img_dir_path}/{seed}_{descriptive_token}_{prompt_suffix}"
                    f"_{'correct' if pred_class == config.class_index else 'wrong'}.jpg"
                )
                #protpro.show_cross_attention(pipe, [prompt],controller, res=16, from_where=("up", "down"),path=img_dir_path+'/'+str(seed))
                utils.numpy_to_pil(image_pt.permute(0, 2, 3, 1).cpu().detach().numpy())[
                    0
                ].save(img_path, "JPEG")
            else:
                img_path = (
                    f"{img_dir_path}/{seed}_{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                )
                #protpro.show_cross_attention(pipe,[prompt],controller, res=16, from_where=("up", "down"),path=img_dir_path+'/'+str(seed)+'_'+str(config.strength)+'_optimized_token_')
                utils.numpy_to_pil(images.permute(0, 2, 3, 1).cpu().detach().numpy())[
                    0
                ].save(dataset_path+'/'+str(seed)+'.jpg', "JPEG")

            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )
        break # Sc token 없을 때는 그냥 EVALUATe 안하겠다. 

def control_strength(config: RunConfig):
    to_tensor = torchvision.transforms.ToTensor()
    
    class_index = config.class_index - 1

    classification_model = utils.prepare_classifier(config)
    classification_model = classification_model.cuda()
    classification_model.eval()

    if "inet" in config.classifier:
        IDX2NAME = IDX2NAME_INET
    elif 'pacs' in config.classifier:
        IDX2NAME = pacs_classes.IDX2NAME
    elif 'cifar10_resnet34' == config.classifier:
        IDX2NAME = cifar10_classes.IDX2NAME
    elif 'cifar100_resnet34' == config.classifier:
        IDX2NAME = cifar100_classes.IDX2NAME
    else:
        IDX2NAME = classification_model.config.id2label

    class_name = IDX2NAME[class_index].split(",")[0]

    exp_identifier = (
        f'{config.exp_id}_{"2.1" if config.sd_2_1 else "1.4"}_{config.epoch_size}_{config.lr}_'
        f"{config.seed}_{config.number_of_prompts}_{config.early_stopping}"
    )
    # Stable model
    token_dir_path = f"token/{class_name}"
    Path(token_dir_path).mkdir(parents=True, exist_ok=True)
    pipe_path = f"pipeline_{token_dir_path}/{exp_identifier}_{class_name}"
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(config.device)

    weight_dtype = torch.float32
    if config.mixed_precision == "fp16":
        weight_dtype = torch.float16

    tokens_to_try = [config.placeholder_token]
    prompt_suffix = " ".join(class_name.lower().split("_"))

    img_dir_path=f"img/{config.prefix}*{config.grad_scale}r_grad{config.lambda_bn}_{class_name}/eval"

    for descriptive_token in tokens_to_try:
        correct = 0
        prompt = f"A {descriptive_token} {prompt_suffix}"
        for seed in range(int(config.test_size/100)):
            generator = torch.Generator(
                device=config.device
            )  # Seed generator to create the inital latent noise
            generator.manual_seed(seed)

            if descriptive_token == config.placeholder_token:
                prompts = [prompt]*2
                print(f"Evaluation for the prompt: {prompt}")
                ### pay 3 times more attention to the word "smiling"
                equalizer = protpro.get_equalizer(pipe, prompts[1], (descriptive_token,), (config.strength,)) #default strength = 5.0
                controller = protpro.AttentionReweight(prompts, pipe, config.num_of_SD_inference_steps, cross_replace_steps=.8,
                                            self_replace_steps=.4,
                                            equalizer=equalizer,
                                            local_blend = None,
                                            controller = None)
                images, x_t = protpro.run_and_display(pipe, prompts, controller, latent=None, run_baseline=False, generator=generator,classification_model=classification_model)
                
                image_pt,image_np = images
                ptp_utils.view_images(image_np, num_rows=1, offset_ratio=0.02, prefix=img_dir_path,postfix=f"/{seed}_strength_{config.strength}_img")
            
            images = torch.nn.functional.interpolate(image_pt, size=224).to(config.device)
            if  config.classifier=="inet_resnet34" or 'pacs' in config.classifier:
                output = classification_model(images)
            else:
                output = classification_model(images).logits
            pred_class = torch.argmax(output[1]).item()

            if descriptive_token == config.initializer_token:
                protpro.show_cross_attention(pipe,[prompt],controller, res=16, from_where=("up", "down"),path=img_dir_path+'/'+str(seed)+'_token')
            else:
                #img_path = (
                #    f"{img_dir_path}/{seed}_{exp_identifier}_{IDX2NAME[pred_class]}.jpg"
                #)
                protpro.show_cross_attention(pipe,[prompt],controller, res=16, from_where=("up", "down"),path=img_dir_path+'/'+str(seed)+'_'+str(config.strength)+'_optimized_token_')

            #utils.numpy_to_pil(image_pt.permute(0, 2, 3, 1).cpu().detach().numpy())[
            #    0
            #].save(img_path, "JPEG")

            if pred_class == class_index:
                correct += 1
            print(f"Image class: {IDX2NAME[pred_class]}")
        acc = correct / config.test_size
        print(
            f"-----------------------Accuracy {descriptive_token} {acc}-----------------------------"
        )


if __name__ == "__main__":
    config = pyrallis.parse(config_class=RunConfig)

    # Check the arguments
    if config.train:
        train(config)
    if config.evaluate:
        evaluate(config)
    if config.control_strength:
        control_strength(config)
