import numpy as np
from collections import deque

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import torch
from torchvision import transforms as T

from core.network import DDQN
from core.config import FRAME_SKIP, FRAME_STACK

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent:
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize models from different checkpoints
        self.models = self.__load_all_models()
        # State management
        self.frames = deque(maxlen=FRAME_STACK)
        self.skip_count = 0
        self.last_action = None
        self.initial_setup_done = False
        # Transform pipeline
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(),
            T.Resize((84, 90)),
            T.ToTensor(),
        ])

    def act(self, observation):
        # Transform raw image into a processable format
        preprocessed_image = self.transform(observation).squeeze(0).numpy()

        if not self.initial_setup_done:
            self.__initialize_frame_buffer(preprocessed_image)

        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action
        self.skip_count = FRAME_SKIP - 1

        self.frames.append(preprocessed_image)
        state_tensor = torch.from_numpy(np.stack(self.frames, axis=0)).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Get action from models
        q_values = self.__compute_q_values(state_tensor)
        action = np.argsort(q_values)[::-1][0]
        self.last_action = action
        return action
    
    def __load_all_models(self):
        models = []
        model_paths = ['icm_ckpt6.pth', 'icm_ckpt7.pth', 'icm_ckpt8.pth']

        for model_path in model_paths:
            model = DDQN(n_channels=4, n_actions=len(COMPLEX_MOVEMENT))
            ckpt = torch.load(model_path, map_location=self.device)
            model.load_state_dict(ckpt)
            model.to(self.device).eval()
            models.append(model)
        return models
    
    def __initialize_frame_buffer(self, initial_frame):
        self.frames.clear()
        for _ in range(FRAME_STACK):
            self.frames.append(initial_frame)
        self.initial_setup_done = True

    def __compute_q_values(self, state_tensor):
        q_values_list = []
        for model in self.models:
            with torch.no_grad():
                q_values = model(state_tensor).squeeze().cpu().numpy()
                q_values_list.append(q_values)
        return np.mean(q_values_list, axis=0)