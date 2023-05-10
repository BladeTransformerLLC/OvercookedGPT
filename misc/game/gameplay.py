# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime
from typing import Tuple
from multiprocessing.connection import Client
import time


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)

        self.client = None

        self.current_frame: int = 0
        self.current_agent_id: int = 1
        self.action_str = None
        self.action_loc = None


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
            self.client.close()
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return

            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent_id = int(pygame.key.name(event.key))
                    self.current_agent = self.sim_agents[self.current_agent_id-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                #print("--------------------------------------------------------")
                #print(f"GamePlay.current_agent_id = {self.current_agent_id}")
                #print(f"GamePlay.current_agent.location = {self.current_agent.location}")
                #print(f"GamePlay.current_agent.action = {action}")
                self.action_str, self.action_loc = interact(self.current_agent, self.world)


    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            self.__send_state()
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            self.current_frame += 1
        self.on_cleanup()


    def __get_agent_location(self, id=0) -> Tuple[int, int]:
        assert 0 <= id < len(self.sim_agents), f"id out of bounds: {id}"
        return self.sim_agents[id].location


    def __send_state(self):
        loc: Tuple[int, int] = self.__get_agent_location(self.current_agent_id-1)
        try:
            if self.client is None:
                self.client = Client(("localhost", 6000))
            self.client.send([self.current_frame, self.current_agent_id, loc, self.action_str, self.action_loc])
        except Exception as e:
            print(f"ERROR: Exception in self.client.send(): {e}")
            self.client = None


