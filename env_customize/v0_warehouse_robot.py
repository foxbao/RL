import random
from enum import Enum

class RobotAction(Enum):
    LEFT=0
    DOWN=1
    RIGHT=2
    UP=3
    
class GridTile(Enum):
    _FLOOR=0
    ROBOT=1
    TARGET=2
    
    def __str__(self):
        return self.name[:1]
    
class WarehouseRobot:
    def __init__(self, grid_rows=4,grid_cols=5):
        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.reset()
        
    def reset(self,seed=None):
        self.robot_pos=[0,0]
        random.seed(seed)
        self.target_pos=[
            random.randint(1,self.grid_rows-1),
            random.randint(1,self.grid_cols-1)
        ]
        
    def perform_action(self, robot_action:RobotAction) -> bool:
        self.last_action = robot_action

        # Move Robot to the next cell
        if robot_action == RobotAction.LEFT:
            if self.robot_pos[1]>0:
                self.robot_pos[1]-=1
        elif robot_action == RobotAction.RIGHT:
            if self.robot_pos[1]<self.grid_cols-1:
                self.robot_pos[1]+=1
        elif robot_action == RobotAction.UP:
            if self.robot_pos[0]>0:
                self.robot_pos[0]-=1
        elif robot_action == RobotAction.DOWN:
            if self.robot_pos[0]<self.grid_rows-1:
                self.robot_pos[0]+=1

        # Return true if Robot reaches Target
        return self.robot_pos == self.target_pos
    
    def render(self):
        # Print current state on console
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):

                if([r,c] == self.robot_pos):
                    print(GridTile.ROBOT, end=' ')
                elif([r,c] == self.target_pos):
                    print(GridTile.TARGET, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print() # new line
        print() # new line

        # self._process_events()

        # # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        # self.window_surface.fill((255,255,255))

        # # Print current state on console
        # for r in range(self.grid_rows):
        #     for c in range(self.grid_cols):
                
        #         # Draw floor
        #         pos = (c * self.cell_width, r * self.cell_height)
        #         self.window_surface.blit(self.floor_img, pos)

        #         if([r,c] == self.target_pos):
        #             # Draw target
        #             self.window_surface.blit(self.goal_img, pos)

        #         if([r,c] == self.robot_pos):
        #             # Draw robot
        #             self.window_surface.blit(self.robot_img, pos)
                
        # text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        # text_pos = (0, self.window_size[1] - self.action_info_height)
        # self.window_surface.blit(text_img, text_pos)       

        # pygame.display.update()
                
        # # Limit frames per second
        # self.clock.tick(self.fps)  
        
        
# For unit testing
if __name__=="__main__":
    warehouseRobot = WarehouseRobot()
    warehouseRobot.render()

    # while(True):
    for i in range(25):
        rand_action = random.choice(list(RobotAction))
        print(rand_action)

        warehouseRobot.perform_action(rand_action)
        warehouseRobot.render()