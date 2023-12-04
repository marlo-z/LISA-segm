from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

import re


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class LisaMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaMetaModel, self).__init__(config)

        # print("DEBUG: in init LisaMetaModel")
        # print("config has train_mask_decoder:", hasattr(self.config, "train_mask_decoder"))
        # print("config:", self.config)
        # input()

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):      # this is false for both train, val
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_lisa_modules(self.config)

    # this is called in train_ds.py, 
    def initialize_lisa_modules(self, config):
        # SAM
        # print("DEBUG: initialize_lisa_modules")
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True
        
        # model.visual_model = SAM VIT-h --> used for encoding image into embeddings 
        # that will be used for mask decoder together with <seg> token embeddings to generate segmentation mask

        # Projection layer
        # ---> this replaces the "lm_head" that maps hidden_state to vocab_size?
        in_dim = config.hidden_size         # dimension of LLM?
        out_dim = config.out_dim            # vocab_size?
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        # Question: 
        # Super class "LlavaLlamaForCausalLm": __init__() method sets:
        # self.model = LlavaLlamaModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        ### Un-used ###
        # Box projector: MLP layer projecting box embedding dimension --> transformer dimension
        # box_proj_params = kwargs.pop("box_projector_params")
        # box_embed_dim = kwargs.pop("box_embed_dim")
        # self.initialize_box_projector(box_proj_params, box_embed_dim)

        # Initialize weights and apply final processing
        self.post_init()
    

    ### Useless now, used project DINO last layer bbox embeding to transformer dimension ###
    def initialize_box_projector(self, params, in_dim):
        out_dim = 5120          # dimension of transformer

        if params == 'linear':
            return nn.Linear(in_dim, out_dim)

        match = re.match(r'mlp_(\d+)x_(\d+)l_(\w+)$', params)
        if match:
            hidden_dim_ratio = int(match.group(1))
            hidden_dim = in_dim * hidden_dim_ratio
            num_layers = int(match.group(2))
            activation = match.group(3)

            if activation == "gelu":
                activation = nn.GELU()
            elif activation == "relu":
                activation == nn.RELU()
            else:
                raise ValueError(f'Unknown activation function: {activation}')

            modules = [nn.Linear(in_dim, hidden_dim), activation]
            for _ in range(1, num_layers):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(activation)
            # final layer
            modules.append(nn.Linear(hidden_dim, out_dim))

            # print("DEBUG:", "Box Projection layer consists:", modules)
            self.box_projector = nn.Sequential(*modules)

            # Question: neccesary?
            self.box_projector.train()
            for param in self.box_projector.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f'Unknown box projector parameters: {params}')

    # Use SAM (visual_model) --> embedding of image (used later for mask decoder)
    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    # LISA's main forward
    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,         # Question: what is images_clip vs images?
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        cropped_boxes: torch.FloatTensor = None, 
        inference: bool = False,
        **kwargs,
    ):
        # images:       torch.Size([2, 3, 1024, 1024])
        # images clip:  torch.Size([2, 3, 224, 224])
        #                          [n_images, c, h, w]
        # input_ids:    torch.Size([6, 186])
        #                          [n_sentences, n_tokens]
        # offset:       [0, 3, 6]  (multiple Q-A pair correspond to same image)
        # cropped_boxes:   [n_images, n_boxes, 3, 224, 224]

        # DEBUG: weights data type, somehow got converted to bfloat16 ???
        # (use args.precision = 'bf16' by default)
        # for name, param in self.box_projector.named_parameters():
        #     print(name, param.dtype)
        # input()

        # TODO: this is a temp fix (need to use args.precision)
        # box_embeds = box_embeds.bfloat16()
        # box_embeds.cuda()
        # box_embeds = self.box_projector(box_embeds)
        # n_boxes = box_embeds.size(1)
        # box_embeds: [n_images, n_boxes, transformer_dim (5120)]

        # Temp
        n_boxes = cropped_boxes.size(1)

        # print("n boxes:", n_boxes)
        # print("n text tokens:", input_ids.size())

        # image_embeddings = SAM_encoder(images) --> embeddings fed into decoder to generate output mask
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        # seg_token_mask used as an index mask later
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx         # (omit first token)

        # seg_token_mask 1: torch.Size([6, 186])  (bool)

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),    # (append 1 token at the end)
            ],
            dim=1,
        )
    
        # concatenate [seg_token_mask:[6, 186], zeros[6, 1]]
        # seg_token_mask 2: torch.Size([6, 187])      (bool)

        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255 + n_boxes)).bool().cuda(), seg_token_mask],
            dim=1,
        )

        ### Since the <img> token will be replaced by 256 clip generated image embeddings 
        ### Therefore, need to append 255 clip tokens + n box tokens
        # concatenate [zeros[6, 255 + 100(n_boxes)], seg_token_mask:[6, 187]]
        # seg_token_mask3 : torch.Size([6, 542])

        if inference:
            n_batch = 1
            length = input_ids.shape[0]

            # print("input ids", input_ids.shape)             # [5, 64]    (n_txt, n_tokens)
            # print("images clip", images_clip.shape)         # [1_img, 3, 224, 224]
            # print("cropped boxes", cropped_boxes.shape)     # [1_img, n_boxes, 3, 224, 224]
            # input()

            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()
            cropped_boxes_extend = cropped_boxes.expand(length, -1, -1, -1, -1).contiguous()

            # print("after expanding ")
            # print("images clip extend", images_clip_extend.shape)         # [n_txt, 3, 224, 224]
            # print("cropped boxes extend", cropped_boxes_extend.shape)     # [n_txt, n_boxes, 3, 224, 224]

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    # Added:
                    cropped_boxes = cropped_boxes_extend[: end_i - start_i],
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            '''expanding batch size to match number of input text'''
            images_clip_list = []
            for i in range(len(offset) - 1):
                # offset measures num text inputs corresponding to this image
                # len(offset) = original number of images
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)                           # add 1 dimension in front
                    .expand(end_i - start_i, -1, -1, -1)    # duplicates k number of times in the new dimension
                    .contiguous()
                )
                # each images_clip[i] expanded from [3, 224, 224] to [3, 3, 224, 224]
                #   unsqueeze: [3, 224, 224] --> [1, 3, 224, 224]
                #   expand:    [1, 3, 224, 224] --> [k, 3, 224, 224]
                #   concat:    list[(k, 3, 224, 224) x num_imgs] ---> [k * num_imgs, 3, 224, 224]      
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
            # So now each images_clip[i] corresponds to 1 text input (previously corresponded to multiple)

            # Final images clip torch.Size([6, 3, 224, 224])  (batch size went from 2 --> 6)

            '''expanding batch size to match number of input text'''
            expanded_boxes_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                boxes_i = (
                    cropped_boxes[i]
                    .unsqueeze(0)                                 # add extra dimension in front
                    .expand(end_i - start_i, -1, -1, -1, -1)      # duplicate in that added dimension
                    .contiguous()
                )
                expanded_boxes_list.append(boxes_i)
            
            expanded_boxes_list = torch.cat(expanded_boxes_list, dim=0)

            # boxes_i (for i-th image in batch):        [n_boxes, 3, 224, 224]
            # unsqueeze:                                [1, n_boxes, 3, 224, 224]
            # expand:                                   [k_i, n_boxes, 3, 224, 224]
            # concat:                                   [k1+...+kn, n_boxes, 3, 224, 224] where k1+...+kn = num of input texts
            # final expanded_boxes_list:                [text_bs, n_boxes, 3, 224, 224]
            # NOTE: each image's boxes will be will be duplicated k_i times, placed contiguously in the final tensor
            #       (as if there were k_i identical images with the exact same bboxes)

            # print("Image clip after expanding:", images_clip.size())
            # print("Original boxes:", cropped_boxes.size())
            # print("Boxes after expanding:", expanded_boxes_list.size())
            # print()
            # input()

            cropped_boxes = expanded_boxes_list

            # LlavaLlamaForCausalLM --> forward()
            output = super().forward(
                cropped_boxes = cropped_boxes, 
                images=images_clip,                 # input images: images_clip dimension:
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
        
        # output hidden states
        # [torch.Size([6, 420, 5120]), torch.Size([6, 420, 5120]), torch.Size([6, 420, 5120])]

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        ### Mask decoding process

        # Problem: dimension doesn't match, after adding box tokens
        # last_hidden_state tensor has dimensions [6, (text + image + box) tokens, 256]
        # but seg_token_mask as dimensions [6, (text + image) tokens]
        # dimension doesn't match, after adding box tokens

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        # Question
        pred_embeddings = last_hidden_state[seg_token_mask]             # for extracting the embedding of the <seg> token
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
