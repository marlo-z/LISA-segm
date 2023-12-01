#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

# from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX)

from .multimodal_encoder.builder import build_vision_tower


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        # print("LLAVA META MODEL INIT:")
        # print("Build vision tower:")
        # print("Config:", config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            # build_vision_tower: --> creates CLIPVisionTower (custom class)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        vision_tower = build_vision_tower(model_args)       # Question: creates vision_tower again? (Already created in __init__)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                self.config.mm_hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # Encode image into image embeddings --> project to text embedding space/dimension
    # model.vision_tower --> CLIP, generates image embeddings to be fed into LLM (different from model.visual_model)
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        '''
        Vision Tower: CLIPVisionModel Encoder
        MM Projector: Linear(1024, 5120)
        1) input images [n, 3, 224, 224]
        2) CLIP encode features [n, 256, 1024] (patch features)
        3) projection [n, 256, 5120]
        '''
        # print("vision_tower:", self.get_model().get_vision_tower())
        # print("mm projector:", self.get_model().mm_projector)
        # print("images:", images.size())
        # print("image_features:", image_features.size())
        return image_features

    # TODO: add spatial information (bbox coordinate)
    # mlp(bbox coords) --> [n_boxes, 1024] + add to pooler_output
    # 2 layers
    # normalize bbox coords --> [-1, 1]
    def encode_boxes(self, cropped_boxes):
        '''
        cropped_boxes:      [n_imgs, n_boxes, 3, 224, 224]
        linearize_boxes:    [n_imgs * n_boxes, 3, 224, 224]
        outputs <-- encode(linearize_boxes)
        outputs:            [n_imgs * n_boxes, 256, 1024]
        pooled_features:    [n_imgs * n_boxes, 1024]
        boxes_features:     [n_imgs * n_boxes, 5120]
        (re-batch)          [n_imgs, n_boxes, 5120]
        '''
        n_imgs, n_boxes = cropped_boxes.size(0), cropped_boxes.size(1)
        #print("n_images, n_boxes", n_imgs, n_boxes)

        linearize_boxes = cropped_boxes.view(n_imgs * n_boxes, 3, 224, 224)
        pooled_features = self.get_model().get_vision_tower()(linearize_boxes, pool_features=True)
        box_features = self.get_model().mm_projector(pooled_features)
        box_features = box_features.view(n_imgs, n_boxes, 5120)


        # TODO: extract a mask for identifying which boxes are pads
        '''
        cropped_boxes:       [n_imgs, n_boxes, 3, 224, 224]
        flattened_imgs:      [n_imgs, n_boxes, 3*224*224]
        padding_mask:        [n_imgs, n_boxes]                  padding_mask[i][j] = True, if ith image, jth box is padded all -1's
        box_features:        [n_imgs, n_boxes, 5120]            boxes_features[padding_mask] =set= all -1
        '''
        # extract padding mask
        flattened_imgs = cropped_boxes.view(n_imgs, n_boxes, 3*224*224)
        pad = torch.ones(3*224*224).cuda() * -1
        padding_mask = torch.all(flattened_imgs == pad, dim = 2)
        
        # set embeddings of padded imgs to all -1's  
        new_pad = (torch.ones(5120) * -1).cuda().to(box_features.dtype)
        box_features[padding_mask] = new_pad

        ### check ###
        # positions in box_features where embedding is all -1's matches 
        # positions in cropped_boxs where padded imgs are all -1's
        embeds_padding_mask = torch.all(box_features == new_pad, dim=2)
        # print("cropped_boxes padding mask:", padding_mask.nonzero())
        # print("box_features padding mask:", embeds_padding_mask.nonzero())
        # print("num padded embeds:", len(embeds_padding_mask.nonzero()))
        # assert torch.all(embeds_padding_mask == padding_mask).item() 
        # input()

        # TODO: set the features corresponding to the padded boxes to -1, using mask
        # NOTE: each of the (n_imgs * n_boxes) cropped images will be encoded independently,
        #       therefore, real imgs won't be mixed with padded imgs
        #       however, sometimes ~10 imgs could be padded, thus creating a lot of overhead
        

        # print("input boxes:", cropped_boxes.size())
        # print("reshaped boxes:", linearize_boxes.size())
        # print("pooled features:", pooled_features.size())
        # print("projected box features:", box_features.size())
        # print("reshaped box features:", box_features.size())
        # input()
        return box_features

    ''' 
    Encode input text, images, cropped boxes into embeddings are merge them
    (encode) images           [batch, 3, 224, 224]             -->     image_embeds [batch, 256, 5120]
    (encode) cropped_boxes    [batch, n_boxes, 3, 224, 224]    -->     box_embeds   [batch, n_boxes, 5120]
    (concat) image_features   [batch, img_embeds + box_embeds, 5102]
    (encode) input_ids   -->  text_embeds
    (concat) new_input_embeds [batch, img_embeds + box_embeds + text_embeds, 5210]
    '''
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, cropped_boxes = None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        # Question: what is this if-case for?
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
            # image_features --> image embeddings, encode image using CLIP (model.vision_tower)


        # CLIP encode cropped boxes --> embeds 
        # input:        cropped images [n_imgs, n_boxes, 3, 224, 224]
        # output:       box features   [n_imgs, n_boxes, 5120]
        box_features = self.encode_boxes(cropped_boxes)

        #### Append bbox token embeddings for each image ####
        # original image_features: [n_imgs, n_img_tokens=256, vec_dim=5120]
        # box_features: [n_imgs, n_boxes, embed_dim=5120]
        # Appended image_features: [n_imgs, img_tokens + box_tokens, embed_dim=5120]
        
        
        image_features = torch.cat([image_features, box_features], dim=1)
        # print("before adding box features", image_features.size())
        # print("after adding box features", image_features.size())
        # input()


        # new_input_embeds will contain image embeddings insertted between text embeddings 
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        # iterate over each input sequence of word/token ids (each input text)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # if not <image token> is found
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)         # model.embed_tokens --> text embedings
                cur_input_embeds = (
                    cur_input_embeds
                    + (
                        0.0 * self.get_model().mm_projector(vision_tower.dummy_feature)
                    ).sum()
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # positions of where <image token> appeared
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            # iterate over all positions where <image token appeared>
            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model()
                        .embed_tokens(cur_input_ids[: image_token_start - 1])
                        .detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start - 1 : image_token_start]
                        )
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start : image_token_start + 1]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(
                            cur_input_ids[image_token_start + 1 : image_token_start + 2]
                        )
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_new_labels.append(
                            cur_labels[image_token_start + 1 : image_token_start + 2]
                        )
                        cur_labels = cur_labels[image_token_start + 2 :]
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=labels.device,
                                dtype=labels.dtype,
                            )
                        )
                        cur_labels = cur_labels[image_token_start + 1 :]
                cur_image_idx += 1
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_input_ids = cur_input_ids[image_token_start + 2 :]
                else:
                    cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                    self.config, "mm_use_im_start_end", False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids).detach()
                    )
                elif getattr(self.config, "mm_use_im_start_end", False):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                else:
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids)
                    )
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [
                x.to(device=self.device) for x in cur_new_input_embeds
            ]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        # TODO: update attention_mask
        # attention mask: [batch, n_tokens] --> atten_mask[:, i] = True if i-th token is not a pad token
        # new_input_embeds --> find which tokens is padded (all -1)

        '''
        new_input_embeds:           [batch_size, n_embeds, 5120]      (n_embeds include img_embeds, box_embeds, text_embeds)
        attention_mask:             [batch_size, n_embeds]
        padding_mask:               [batch_size, n_embeds]            (mask[i, j] = True <-> ith example's jth token is a padded box_embed all -1's)
        '''

        # extract padding mask (positions where embeddings of padded boxes are all -1)
        pad = (torch.ones(5120) * -1).cuda()
        padding_mask = torch.all(new_input_embeds == pad, dim = 2)
        # print("final input embeds padding tokens:", len(padding_mask.nonzero()))

        # print("attention mask dim:", attention_mask.shape)
        # print("padding mask dim:", padding_mask.shape)

        ''' 
        update attention:
        if padding_mask[i, j] = True:
              --> input_embeds[i, j] = embed of padded box 
              --> attention_mask[i, j] <- False
        otherwise: 
              --> keep same
        '''
        _false = torch.zeros_like(attention_mask).bool().cuda()
        updated_attention_mask = torch.where(padding_mask, _false, attention_mask)

        ### check ###
        # total_num_tokens = attention_mask.size(0) * attention_mask.size(1)
        # num_pad_tokens = total_num_tokens - len(attention_mask.nonzero())
        # new_num_pad_tokens = total_num_tokens - len(updated_attention_mask.nonzero())
        # num_pad_boxes = len(torch.all(box_features == pad, dim = 2).nonzero())
        # print("previous pad tokens:", num_pad_tokens)
        # print("new pad tokens:", new_num_pad_tokens)
        # print("num pad boxes:", num_pad_boxes)

        return None, updated_attention_mask, past_key_values, new_input_embeds, new_labels

    # def initialize_vision_tokenizer(self, model_args, tokenizer):
    def initialize_vision_tokenizer(self, model_args, num_new_tokens):
        # if model_args.mm_use_im_patch_token:
        #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        #     self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            # num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            # self.resize_token_embeddings(len(tokenizer))

            # if num_new_tokens > 0:
            #     input_embeddings = self.get_input_embeddings().weight.data
            #     output_embeddings = self.get_output_embeddings().weight.data

            #     input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)
            #     output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            #         dim=0, keepdim=True)

            #     input_embeddings[-num_new_tokens:] = input_embeddings_avg
            #     output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
