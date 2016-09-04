"""
This class draws kernels and outputs of each layer
"""
import lasagne
import pygame
from pygame.locals import *
import numpy as np
import theano
import theano.tensor as T
import math

_GAP = 10
SCREEN_INIT_W = 1250
SCREEN_INIT_H = 1250
# SCREEN_INIT_H = 920

# TODO weight sharing, global normalizasyon bar ile featurelarin -6 +12 gibi cizdirilmesi
class VisualInterface(object):
    def __init__(self, network, data_set, ale_agent):
        pygame.init()
        # self.screen = pygame.display.set_mode((1280, 1250), pygame.RESIZABLE)
        self.screen = pygame.display.set_mode((1024, 700), pygame.RESIZABLE)
        pygame.display.set_caption("Deep Q Reinforcement Learning Display")
        self.clock = pygame.time.Clock()
        self.delay = 30
        self.network = network
        self.data_set = data_set
        self.font = pygame.font.SysFont("Arial", 15)
        self.general_offset_y = 0
        self.zero_precision = 0.01
        self.ale_agent = ale_agent

        self.ratio_w = 1
        self.ratio_h = 1

        self.pageNo = 1
        self.pages = [[PageInformation(1, "Layer 1 Global Norm")],
                      [PageInformation(1, "Layer 1 Local Norm", global_norm=False)],
                      [PageInformation(2, "Layer 2", draw_w=False)],
                      # [PageInformation(2, "Layer 2 Local Norm", draw_w=False, global_norm=False)],
                      [PageInformation(3, "Layer 3", draw_w=False)],
                      # [PageInformation(3, "Layer 3 Local Norm", draw_w=False, global_norm=False)],
                      [PageInformation(4, "Layer 4", draw_w=False)],
                      # [PageInformation(4, "Layer 4 Local Norm", draw_w=False, global_norm=False)],
                      [PageInformation(5, "Layer 5", draw_w=False)],
                      [PageInformation(1, "Layer 1 ", draw_w=False),
                       PageInformation(2, "Layer 2 ", draw_w=False),
                       PageInformation(3, "Layer 3 ", draw_w=False),
                       PageInformation(4, "Layer 4 ", draw_w=False),
                       PageInformation(5, "Layer 5 ", draw_w=False)]]

        self.q_layers = lasagne.layers.get_all_layers(self.network.l_out)

        states = T.tensor4('states')
        # layer1_outputs = lasagne.layers.get_output(
        #     self.q_layers[1],
        #     states / 255.)
        #
        # layer2_outputs = lasagne.layers.get_output(
        #     self.q_layers[2],
        #     states / 255.)
        #
        # layer3_outputs = lasagne.layers.get_output(
        #     self.q_layers[3],
        #     states / 255.)
        #
        # layer4_outputs = lasagne.layers.get_output(
        #     self.q_layers[4],
        #     states / 255.)

        layer_outs = [None] * len(self.q_layers)
        for i in range(0, len(self.q_layers)):
            layer_outs[i] = lasagne.layers.get_output(self.q_layers[i], states / 255.)


        self.fn_layer123_out = theano.function([], layer_outs, givens={states: self.network.states_shared})

        # self.fn_layer123_out = theano.function([], [layer1_outputs,
        #                                             layer2_outputs,
        #                                             layer3_outputs,
        #                                             layer4_outputs],
        #                                givens={states: self.network.states_shared})

    def delay_vis(self):
        if self.delay >= 0:
            self.clock.tick(self.delay)

    def update_vis(self):
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.delay -= 5
                if self.delay <= 1:
                    self.delay = 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.delay += 5
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.general_offset_y -= 10
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.general_offset_y += 10
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                self.zero_precision /= 2
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                self.zero_precision *= 2
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.pageNo -= 1
                if self.pageNo == 0:
                    self.pageNo = len(self.pages)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.pageNo = self.pageNo % len(self.pages) + 1
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                pygame.display.flip()
            return False

    def draw_layer_weights(self, offset_x, offset_y, weights, gap, global_norm):
        ox = offset_x
        oy = offset_y
        for i in range(0, weights.shape[0]):
            w = weights[i, 0, :, :]
            w = self.normalizeBy(w, weights.min(), weights.max())
            w = self.to_RGB_greyscale(w)

            # 4 filters per block
            block_size = weights.shape[1]

            surf = self.create_surface(w)
            block_multiplier = math.ceil(math.sqrt(block_size))
            block_width = int((surf.get_width()*5 + gap) * block_multiplier * self.ratio_w)
            block_height = int((surf.get_height()*5 + gap) * block_multiplier * self.ratio_h)
            if ox + block_width > self.screen.get_width():
                ox = offset_x
                oy += + block_height + gap

            for j in range(0, weights.shape[1]):
                w = weights[i, j, :, :]
                if global_norm:
                    # w = self.normalizeBy(w, weights.min(), weights.max())
                    w = self.norm_RGB(w, weights.min(), weights.max())
                else:
                    w = self.normalize(w)
                    # w = self.normalizeBy(w, weights.min(), weights.max())
                    w = self.to_RGB_greyscale(w)
                surf = self.create_surface(w)
                width = int(surf.get_width()*5 * self.ratio_w)
                height = int(surf.get_height()*5 * self.ratio_h)
                ox_ = ox + (width + gap) * (j % math.ceil(math.sqrt(block_size)))
                oy_ = oy + (height + gap) * math.floor(j / math.ceil(math.sqrt(block_size)))
                self.screen.blit(pygame.transform.scale(surf, (width, height)), (ox_, oy_))
            ox += block_width + gap
        return oy + block_height

    def draw_layer_activations(self, offset_x, offset_y, outs, gap, global_norm):
        dense_layer = len(outs.shape) == 1
        if dense_layer:
            return self.draw_dense_activations(offset_x, offset_y, outs, gap, global_norm)
        else:
            return self.draw_conv_activations(offset_x, offset_y, outs, gap, global_norm)

    def draw_dense_activations(self, offset_x, offset_y, outs, gap, global_norm):
        ox = offset_x
        oy = offset_y
        if global_norm:
            norm_outs = self.normalizeBy(outs, outs.min(), outs.max())
        else:
            norm_outs = self.normalize(outs)
        col = int(self.screen.get_width() / 25)
        row = int(math.ceil(float(len(outs)) / col))
        out0 = np.zeros(row * col, outs.dtype)
        out0[:len(outs)] = norm_outs
        out0 = np.reshape(out0, (row, col))
        out0 = self.to_RGB_greyscale(out0)

        surf = self.create_surface(out0)
        width = int(surf.get_width()*20 * self.ratio_w)
        height = int(surf.get_height()*20 * self.ratio_h)

        self.screen.blit(pygame.transform.scale(surf, (width, height)), (ox, oy))
        return oy + height

    def draw_conv_activations(self, offset_x, offset_y, outs, gap, global_norm):
        ox = offset_x
        oy = offset_y
        for i in range(0, outs.shape[0]):
            out0 = outs[i]
            if global_norm:
                out0 = self.normalizeBy(out0, outs.min(), outs.max())
            else:
                out0 = self.normalize(out0)
            out0 = self.to_RGB_greyscale(out0)

            surf = self.create_surface(out0)
            width = int(surf.get_width()*5 * self.ratio_w)
            height = int(surf.get_height()*5 * self.ratio_h)
            if ox + width > self.screen.get_width():
                ox = offset_x
                oy += + height + gap

            self.screen.blit(pygame.transform.scale(surf, (width, height)), (ox, oy))
            ox += width + gap
        return oy + height

    def draw_info_panel(self, offset_x, offset_y, step, weights, outs, gap):
        gap = gap / 2
        # render step coutner
        text = self.font.render("Step: " + str(step) + "   FPS: " + str(self.delay) + "   Reward: " + str(self.ale_agent.episode_reward), True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        # render page
        text = self.font.render("Page: " + str(self.pageNo) + " of " + str(len(self.pages)) + "  " + self.pages[self.pageNo-1][0].definition, True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        # layer 1 weights are normalized by
        text = self.font.render("weight min-max: " + str(weights.min()) + " " + str(weights.max()), True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        diff = weights.max() - weights.min()
        zero_interval = diff * self.zero_precision / 2
        text = self.font.render("Zero interval: +-" + str(zero_interval), True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        p = zero_interval / (weights.max() - weights.min()) * 255
        red = math.fabs(math.fabs(weights.min()) / (weights.max() - weights.min()) * 255)
        text = self.font.render("Zero RGB: " + str(red) + " +-" + str(p), True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        # layer 1 outs are normalized by
        text = self.font.render("output min-max: " + str(outs.min()) + " " + str(outs.max()), True, (255, 255, 255))
        self.screen.blit(text, (0, offset_y))
        offset_y += text.get_height() + gap

        return offset_y

    def draw(self, step, original_img):
        self.delay_vis()
        pygame.event.pump()
        self.screen.fill((0, 0, 67))  # clear screen
        self.ratio_w = self.screen.get_width() / float(SCREEN_INIT_W)
        self.ratio_h = self.screen.get_height() / float(SCREEN_INIT_H)

        gap = int(self.ratio_w * _GAP)

        four_img = self.data_set.last_phi()  # this should include current image too.

        self.set_shared_var(four_img, self.network)
        all_layer_outs = self.fn_layer123_out()

        # Original image
        greyscale = self.to_RGB_greyscale(four_img[0])
        surf = self.create_surface(original_img)

        width = int(four_img[0].shape[0]*4 * self.ratio_w)
        height = int(four_img[0].shape[1]*4 * self.ratio_h)
        self.screen.blit(pygame.transform.scale(surf, (width, height)), (0, self.general_offset_y))

        greyscale = self.to_RGB_greyscale(four_img[four_img.shape[0]-1])
        surf = self.create_surface(greyscale)

        width = int(surf.get_width()*4 * self.ratio_w)
        height = int(surf.get_height()*4 * self.ratio_h)
        self.screen.blit(pygame.transform.scale(surf, (width, height)), (0, self.general_offset_y + height + 10))

        this_page = self.pages[self.pageNo - 1]

        offset_x = width + gap
        info_offset_y = self.general_offset_y + 2*height + gap
        offset_y = self.general_offset_y

        for i in range(0, len(this_page)):
            global_norm = this_page[i].global_norm
            draw_weights = this_page[i].draw_weights
            draw_activations = this_page[i].draw_activations
            layer_index = this_page[i].layer_index

            # layer outputs and weights
            outs = all_layer_outs[layer_index][0]
            weights = self.q_layers[layer_index].W.get_value()

            info_offset_y = self.draw_info_panel(0, info_offset_y, step, weights, outs, gap)

            if draw_weights:
                offset_y = self.draw_layer_weights(offset_x, offset_y, weights, gap, global_norm)

            if draw_activations:
                offset_y = self.draw_layer_activations(offset_x, offset_y, outs, gap, global_norm)

            offset_y += 2*gap

        pygame.display.flip()
        self.update_vis()

    def to_RGB_greyscale(self, img):
        """
        Change NxM image to NxMx3.
        :param img: greyscale image. N by M
        :return: greyscale image N x M x3
        """
        (width, height) = img.shape[0:2]
        greyscale = np.empty((width, height, 3), dtype=np.uint8)

        # transform 1ch greyscale to 3ch
        greyscale[:, :, 0] = img
        greyscale[:, :, 1] = img
        greyscale[:, :, 2] = img
        return greyscale

    def normalize(self, img):
        n = img - img.min()
        if n.max() == 0:
            n *= 0
        else:
            n = n / n.max()
        return n * 255

    def normalizeBy(self, img, minn, maxx):
        n = img - minn
        global_max = maxx - minn
        if global_max == 0:
            n *= 0
        else:
            n = n / global_max
        return n * 255

    def norm_RGB(self, img, minn, maxx):
        diff = maxx - minn
        zero_interval = diff * self.zero_precision / 2
        c1 = img > -zero_interval
        c2 = img < zero_interval
        c = c1 & c2
        norm_img = self.normalizeBy(img, minn, maxx)
        rgb = self.to_RGB_greyscale(norm_img)
        rgb[c, 0] = 255
        rgb[c, 1] = 0
        rgb[c, 2] = 0
        return rgb

    def to_float32(self, img):
        f = np.empty(img.shap, dtype=np.float32)
        f[:] = img
        return f

    def to_frame_img(self, img):
        (width, height) = img.shape[0:2]
        f = np.empty((4, width, height), dtype=np.uint8)
        f[:] = img
        return f

    def set_shared_var(self, state, network):
        states = np.zeros(self.q_layers[0].shape, dtype=theano.config.floatX)
        states[0, ...] = state
        network.states_shared.set_value(states)

    def create_surface(self, img):
        """
        :param img: RGB image
        :return: flipped & rotated image.
        """
        surf = pygame.surfarray.make_surface(img)
        surf = pygame.transform.rotate(surf, 90)
        surf = pygame.transform.flip(surf, False,True)
        return surf


class PageInformation(object):
    def __init__(self, layer_index, definition, draw_w=True, draw_out=True, global_norm=True):
        self.layer_index = layer_index
        self.definition = definition
        self.draw_weights = draw_w
        self.draw_activations = draw_out
        self.global_norm = global_norm
